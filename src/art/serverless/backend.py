import asyncio
from typing import TYPE_CHECKING, AsyncIterator, Literal

from openai._types import NOT_GIVEN
from tqdm import auto as tqdm

from art.client import Client, ExperimentalTrainingConfig
from art.utils.deploy_model import LoRADeploymentJob, LoRADeploymentProvider

from .. import dev
from ..backend import Backend
from ..trajectories import TrajectoryGroup
from ..types import TrainConfig

if TYPE_CHECKING:
    from ..model import Model, TrainableModel


class ServerlessBackend(Backend):
    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        client = Client(api_key=api_key, base_url=base_url)
        super().__init__(base_url=str(client.base_url))
        self._client = client

    async def close(self) -> None:
        await self._client.close()

    async def register(
        self,
        model: "Model",
    ) -> None:
        """
        Registers a model with the Backend for logging and/or training.

        Args:
            model: An art.Model instance.
        """
        from art import TrainableModel

        if not isinstance(model, TrainableModel):
            print(
                "Registering a non-trainable model with the Serverless backend is not supported."
            )
            return
        client_model = await self._client.models.create(
            entity=model.entity,
            project=model.project,
            name=model.name,
            base_model=model.base_model,
            return_existing=True,
        )
        model.id = client_model.id
        model.entity = client_model.entity

    def _model_inference_name(self, model: "TrainableModel") -> str:
        assert model.entity is not None, "Model entity is required"
        return f"wandb-artifact:///{model.entity}/{model.project}/{model.name}"

    async def _get_step(self, model: "Model") -> int:
        if model.trainable:
            assert model.id is not None, "Model ID is required"
            async for checkpoint in self._client.models.checkpoints.list(
                limit=1, order="desc", model_id=model.id
            ):
                return checkpoint.step
        # Non-trainable models do not have checkpoints/steps; default to 0
        return 0

    async def _delete_checkpoints(
        self,
        model: "TrainableModel",
        benchmark: str,
        benchmark_smoothing: float,
    ) -> None:
        # TODO: potentially implement benchmark smoothing
        assert model.id is not None, "Model ID is required"
        benchmark_values: dict[int, float] = {}
        async for checkpoint in self._client.models.checkpoints.list(model_id=model.id):
            benchmark_values[checkpoint.step] = checkpoint.metrics.get(
                benchmark, -float("inf")
            )
        max_step = max(benchmark_values.keys())
        max_benchmark_value = max(benchmark_values.values())
        if steps_to_delete := [
            step
            for step, benchmark_value in benchmark_values.items()
            if step != max_step and benchmark_value != max_benchmark_value
        ]:
            await self._client.models.checkpoints.delete(
                model_id=model.id,
                steps=steps_to_delete,
            )

    async def _prepare_backend_for_training(
        self,
        model: "TrainableModel",
        config: dev.OpenAIServerConfig | None,
    ) -> tuple[str, str]:
        return str(self._base_url), self._client.api_key

    async def _log(
        self,
        model: "Model",
        trajectory_groups: list[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        # TODO: log trajectories to local file system?
        if not model.trainable:
            print(f"Model {model.name} is not trainable; skipping logging.")
            return
        assert model.id is not None, "Model ID is required"
        await self._client.models.log(
            model_id=model.id, trajectory_groups=trajectory_groups, split=split
        )

    async def _train_model(
        self,
        model: "TrainableModel",
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
        verbose: bool = False,
    ) -> AsyncIterator[dict[str, float]]:
        assert model.id is not None, "Model ID is required"
        training_job = await self._client.training_jobs.create(
            model_id=model.id,
            trajectory_groups=trajectory_groups,
            experimental_config=ExperimentalTrainingConfig(
                learning_rate=config.learning_rate,
                precalculate_logprobs=dev_config.get("precalculate_logprobs"),
            ),
        )
        after: str | None = None
        num_sequences: int | None = None
        pbar: tqdm.tqdm | None = None
        while True:
            await asyncio.sleep(1)
            async for event in self._client.training_jobs.events.list(
                training_job_id=training_job.id, after=after or NOT_GIVEN
            ):
                if event.type == "gradient_step":
                    assert pbar is not None and num_sequences is not None
                    pbar.update(1)
                    pbar.set_postfix(event.data)
                    yield {**event.data, "num_gradient_steps": num_sequences}
                elif event.type == "training_started":
                    num_sequences = event.data["num_sequences"]
                    if pbar is None:
                        pbar = tqdm.tqdm(total=num_sequences, desc="train")
                    continue
                elif event.type == "training_ended":
                    return
                elif event.type == "training_failed":
                    error_message = event.data.get(
                        "error_message", "Training failed with an unknown error"
                    )
                    raise RuntimeError(f"Training job failed: {error_message}")
                after = event.id

    # ------------------------------------------------------------------
    # Experimental support for S3
    # ------------------------------------------------------------------

    async def _experimental_pull_from_s3(
        self,
        model: "Model",
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
        only_step: int | Literal["latest"] | None = None,
    ) -> None:
        raise NotImplementedError

    async def _experimental_push_to_s3(
        self,
        model: "Model",
        *,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
    ) -> None:
        raise NotImplementedError

    async def _experimental_fork_checkpoint(
        self,
        model: "Model",
        from_model: str,
        from_project: str | None = None,
        from_s3_bucket: str | None = None,
        not_after_step: int | None = None,
        verbose: bool = False,
        prefix: str | None = None,
    ) -> None:
        raise NotImplementedError

    async def _experimental_deploy(
        self,
        deploy_to: LoRADeploymentProvider,
        model: "TrainableModel",
        step: int | None = None,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        pull_s3: bool = True,
        wait_for_completion: bool = True,
    ) -> LoRADeploymentJob:
        raise NotImplementedError
