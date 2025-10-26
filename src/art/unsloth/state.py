import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import gc
import unsloth  # type: ignore
from datasets import Dataset
import nest_asyncio
import os
import peft
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.dummy_pt_objects import (
    PreTrainedModel,
    GenerationMixin,
)
from trl import GRPOConfig, GRPOTrainer
from typing import Any, AsyncGenerator, cast, TYPE_CHECKING
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.worker.multi_step_model_runner import MultiStepModelRunner

from ..dev.model import InternalModelConfig

if TYPE_CHECKING:
    from .service import TrainInputs

nest_asyncio.apply()


class CausallLM(PreTrainedModel, GenerationMixin):
    vllm_engine: AsyncLLMEngine


class ModelState:
    """
    A class responsible for initializing and holding references to the model and related state.
    """

    def __init__(self, config: InternalModelConfig) -> None:
        from unsloth_zoo.vllm_rlhf_utils import ColocateWorkerExtension
        from vllm.engine import async_llm_engine

        # Patch MultiStepModelRunner for Unsloth compatibility
        if not hasattr(MultiStepModelRunner, "model"):
            MultiStepModelRunner.model = property(  # type: ignore
                lambda self: self._base_model_runner.model
            )

        # Set effectively unlimited timeout to support engine pausing & resumption
        async_llm_engine.ENGINE_ITERATION_TIMEOUT_S = 2**31 - 1
        # Sticking with V0 engine for now
        os.environ["VLLM_USE_V1"] = "1"
        # We can't use expandable segments with sleep mode
        enable_sleep_mode = config.get("engine_args", {}).get(
            "enable_sleep_mode", False
        )
        if enable_sleep_mode:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
        # Initialize Unsloth model
        # NOTE: We have to patch empty_cache with a no-op during model initialization
        # to avoid an allocator error.
        empty_cache = torch.cuda.empty_cache
        torch.cuda.empty_cache = lambda: None
        from_engine_args = AsyncLLMEngine.from_engine_args

        # NOTE: We also have to patch from_engine_args to control the engine args
        # that are passed to the engine constructor.
        def _from_engine_args(
            engine_args: AsyncEngineArgs, *args: Any, **kwargs: Any
        ) -> AsyncLLMEngine:
            # Get engine args from config
            from dataclasses import fields as _dc_fields
            
            provided = config.get("engine_args", {}) or {}
            accepted_keys = {f.name for f in _dc_fields(type(engine_args))}
            filtered = {k: v for k, v in provided.items() if k in accepted_keys}
            
            # Force disable sleep mode on non-CUDA platforms (ROCm, Intel XPU, etc.)
            # Sleep mode uses CUDA-specific CuMemAllocator and will fail validation on other platforms
            if "enable_sleep_mode" in accepted_keys:
                # Detect platform by checking torch backend capabilities
                # On ROCm: torch.version.hip is set, On CUDA: torch.version.cuda is set
                is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
                is_cuda_native = hasattr(torch.version, 'cuda') and torch.version.cuda is not None and not is_rocm
                
                if not is_cuda_native:
                    # Force disable sleep mode on non-CUDA platforms
                    filtered["enable_sleep_mode"] = False
                    if provided.get("enable_sleep_mode", False):
                        import warnings
                        warnings.warn(
                            f"Sleep mode is not supported on ROCm/HIP platforms. Disabling sleep mode.",
                            UserWarning
                        )
            
            # Add worker extension for in-process model access
            filtered["worker_extension_cls"] = f"{ColocateWorkerExtension.__module__}.{ColocateWorkerExtension.__qualname__}"
            
            return from_engine_args(
                replace(engine_args, **filtered),
                *args,
                **kwargs,
            )

        AsyncLLMEngine.from_engine_args = _from_engine_args
        self.model, self.tokenizer = cast(
            tuple[CausallLM, PreTrainedTokenizerBase],
            unsloth.FastLanguageModel.from_pretrained(**config.get("init_args", {})),
        )
        AsyncLLMEngine.from_engine_args = from_engine_args
        torch.cuda.empty_cache = empty_cache
        torch.cuda.empty_cache()
        
        # Use the filtered enable_sleep_mode value (which is False on non-CUDA platforms)
        actual_sleep_mode = filtered.get("enable_sleep_mode", enable_sleep_mode)
        self.vllm = vLLMState(self.model.vllm_engine, actual_sleep_mode)
        # Initialize PEFT model
        self.peft_model = cast(
            peft.peft_model.PeftModelForCausalLM,
            unsloth.FastLanguageModel.get_peft_model(
                self.model, **config.get("peft_args", {})
            ),
        )
        self.lora_model = cast(peft.tuners.lora.LoraModel, self.peft_model.base_model)
        # Initialize trainer
        data = {"prompt": ""}
        self.trainer = GRPOTrainer(
            model=self.peft_model,  # type: ignore
            reward_funcs=[],
            args=GRPOConfig(**config.get("trainer_args", {})),
            train_dataset=Dataset.from_list([data for _ in range(10_000_000)]),
            processing_class=self.tokenizer,
        )
        self.inputs_queue = asyncio.Queue["TrainInputs"]()

        # Patch trainer _prepare_inputs()
        def _async_prepare_inputs(*_, **__) -> dict[str, torch.Tensor]:
            async def get_inputs() -> "TrainInputs":
                return await self.inputs_queue.get()

            # Force otherwise synchronous _prepare_inputs() to yield
            # with nested asyncio.run() call
            inputs = asyncio.run(get_inputs())

            return cast(dict[str, torch.Tensor], inputs)

        self.trainer._prepare_inputs = _async_prepare_inputs


class vLLMState:
    def __init__(self, async_engine: AsyncLLMEngine, enable_sleep_mode: bool) -> None:
        from ..vllm import (
            create_engine_pause_and_resume_functions,
            patch_allocator,
            patch_lora_request,
            patch_get_lora_tokenizer_async,
            patch_multi_step_model_runner,
        )

        if enable_sleep_mode:
            patch_allocator()
        # Unsloth patches
        patch_lora_request()
        patch_get_lora_tokenizer_async()
        self.async_engine = async_engine
        if enable_sleep_mode:
            self.pause_engine, self.resume_engine = (
                create_engine_pause_and_resume_functions(self.async_engine)
            )
        self.enable_sleep_mode = enable_sleep_mode
        self.driver_worker = cast(
            "WorkerWrapperBase",
            getattr(self.async_engine.engine.model_executor, "driver_worker"),
        )
        if isinstance(self.driver_worker.model_runner, MultiStepModelRunner):
            patch_multi_step_model_runner(self.driver_worker.model_runner)

    @asynccontextmanager
    async def train_mode(self) -> AsyncGenerator[None, None]:
        """
        A context manager pauses the vLLM engine and frees memory for training.
        """
        if not self.enable_sleep_mode:
            yield
            return
        try:
            await self.pause_engine()
            try:
                if self.async_engine.engine.has_unfinished_requests():
                    # Offload KV cache to CPU memory (or disk)
                    await self.async_engine.sleep(level=1)
                else:
                    # Reset prefix cache and discard KV cache
                    await self.async_engine.reset_prefix_cache()
                    await self.async_engine.sleep(level=2)
                free_memory()
                yield
            finally:
                free_memory()
                await asyncio.sleep(0.1)
                await self.async_engine.wake_up()
        finally:
            await self.resume_engine()


def free_memory() -> None:
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
