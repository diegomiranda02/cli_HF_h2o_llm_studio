# Adaptation in the H2O code for publishing models from h2o LLM Studio to Hugging Face using the CLI interface

# CREDITS
 
### H2O company for releasing the h2o LLM Studio software to enable free-of-charge fine-tuning of LLM algorithms

H2O LLM Studio Project: https://h2o.ai/blog/effortless-fine-tuning-of-large-language-models-with-open-source-h2o-llm-studio/

Tutorial on how to use the software using the CLI interface: https://colab.research.google.com/drive/1-OYccyTvmfa3r7cAquw8sioFFPJcn4R9?usp=sharing

### Tomaz Bratanic for sharing a step-by-step guide on how to use the H2O LLM Studio software to do fine-tuning to generate Cypher code

Fine-tuning an LLM model with H2O LLM Studio to generate Cypher statements: https://towardsdatascience.com/fine-tuning-an-llm-model-with-h2o-llm-studio-to-generate-cypher-statements-3f34822ad5

# Perform the steps in the H2o tutorial

1. Clone the h2o LLM Studio repository

Insert ! in front of the command, in case of using Google Colab

```
git clone https://github.com/h2oai/h2o-llmstudio.git
cp -r h2o-llmstudio/. ./
rm -r h2o-llmstudio
```

2. Code to use pipenv

```
# Install pyhon 3.10 that will be used within pipenv
!sudo add-apt-repository ppa:deadsnakes/ppa -y > /dev/null
!sudo apt install python3.10 python3.10-distutils psmisc -y > /dev/null
!curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 > /dev/null

# install requirements
!make setup > /dev/null
```
3. Create the configuration class

```
%%writefile cfg_notebook.py

import os
from dataclasses import dataclass

from llm_studio.python_configs.text_causal_language_modeling_config import ConfigProblemBase, ConfigNLPCausalLMDataset, \
    ConfigNLPCausalLMTokenizer, ConfigNLPAugmentation, ConfigNLPCausalLMArchitecture, ConfigNLPCausalLMTraining, \
    ConfigNLPCausalLMPrediction, ConfigNLPCausalLMEnvironment, ConfigNLPCausalLMLogging
```
If you are using Google Colab use the /content/ dir or the dir you chose to clone the repository

```
ROOT_DIR="/content/"
```

The 'output_directory' is where the model and the files generated from the fine-tuning will be. Every execution of the training is called an experiment, set the variable 'experiment_name' to define a name of the experiment. 'llm_backbone' is the variable to inform which LLM will be fine-tuned.

Set the 'train_dataframe' variable to inform which dataset will be used. 

Dataset suggestion to use: https://github.com/tomasonjo/blog-datasets/tree/main/llm

It's possible to change some configuration parameters. In this case, the epochs parameter was changed from 1 to 15 (epochs=15).

```
@dataclass
class Config(ConfigProblemBase):
    output_directory: str = "output/demo_fine_tuning/" 
    experiment_name: str = "fine_tuning_experiment"
    llm_backbone: str = "EleutherAI/pythia-1.4b-deduped"

    dataset: ConfigNLPCausalLMDataset = ConfigNLPCausalLMDataset(
        train_dataframe=os.path.join(ROOT_DIR, "[PATH TO THE DATASET FILE].csv"),

        validation_strategy="automatic",
        validation_dataframe="",
        validation_size=0.01,

        prompt_column=("instruction",),
        answer_column="output",
        text_prompt_start="",
        text_answer_separator="",

        add_eos_token_to_prompt=True,
        add_eos_token_to_answer=True,
        mask_prompt_labels=False,

    )
    tokenizer: ConfigNLPCausalLMTokenizer = ConfigNLPCausalLMTokenizer(
        max_length_prompt=128,
        max_length_answer=128,
        max_length=256,
        padding_quantile=1.0
    )
    augmentation: ConfigNLPAugmentation = ConfigNLPAugmentation(token_mask_probability=0.0)
    architecture: ConfigNLPCausalLMArchitecture = ConfigNLPCausalLMArchitecture(
        backbone_dtype="float16",
        gradient_checkpointing=False,
        force_embedding_gradients=False,
        intermediate_dropout=0
    )
    training: ConfigNLPCausalLMTraining = ConfigNLPCausalLMTraining(
        loss_function="CrossEntropy",
        optimizer="AdamW",

        learning_rate=0.00015,

        batch_size=4,
        drop_last_batch=True,
        epochs=15,
        schedule="Cosine",
        warmup_epochs=0.0,

        weight_decay=0.0,
        gradient_clip=0.0,
        grad_accumulation=1,

        lora=True,
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules="",

        save_best_checkpoint=False,
        evaluation_epochs=1.0,
        evaluate_before_training=False,
    )
    prediction: ConfigNLPCausalLMPrediction = ConfigNLPCausalLMPrediction(
        metric="BLEU",

        min_length_inference=2,
        max_length_inference=256,
        batch_size_inference=0,

        do_sample=False,
        num_beams=2,
        temperature=0.3,
        repetition_penalty=1.2,
    )
    environment: ConfigNLPCausalLMEnvironment = ConfigNLPCausalLMEnvironment(
        mixed_precision=True,
        number_of_workers=4,
        seed=1
    )

```

4. Create the run.sh script file

```
%%writefile run.sh

pipenv run python train.py -C cfg_notebook.py &

wait
echo "all done"
```

5. Execute the run.sh script to train the model based on the parameters defined previously

```
!sh run.sh
```

# The following code snippets were adapted using existing codes in the h2o LLM Studio project

6. Install the libs required

```
pip install accelerate h2o_wave transformers GPUtil boto3 datasets sqlitedict openai sacrebleu peft coolname bitsandbytes
```

7. Imports

```
from app_utils.sections.chat import load_cfg_model_tokenizer
from llm_studio.src.utils.modeling_utils import check_disk_space
from jinja2 import Environment, FileSystemLoader
import huggingface_hub
import accelerate
import transformers
import torch
import os
```

8. get_model_card used in the publish_to_hugging_face_cli method

```
def get_model_card(cfg, model, repo_id) -> huggingface_hub.ModelCard:
    card_data = huggingface_hub.ModelCardData(
        language="en",
        library_name="transformers",
        tags=["gpt", "llm", "large language model", "h2o-llmstudio"],
    )
    card = huggingface_hub.ModelCard.from_template(
        card_data,
        template_path="model_card_template.md",
        base_model=cfg.llm_backbone,  # will be replaced in template if it exists
        repo_id=repo_id,
        model_architecture=model.backbone.__repr__(),
        config=cfg.__repr__(),
        use_fast=cfg.tokenizer.use_fast,
        min_new_tokens=cfg.prediction.min_length_inference,
        max_new_tokens=cfg.prediction.max_length_inference,
        do_sample=cfg.prediction.do_sample,
        num_beams=cfg.prediction.num_beams,
        temperature=cfg.prediction.temperature,
        repetition_penalty=cfg.prediction.repetition_penalty,
        text_prompt_start=cfg.dataset.text_prompt_start,
        text_answer_separator=cfg.dataset.text_answer_separator,
        trust_remote_code=cfg.environment.trust_remote_code,
        transformers_version=transformers.__version__,
        accelerate_version=accelerate.__version__,
        torch_version=torch.__version__.split("+")[0],
        end_of_sentence=cfg._tokenizer_eos_token
        if cfg.dataset.add_eos_token_to_prompt
        else "",
    )
    return card
```
  
8. Code to publish the model using CLI

```
experiment_path = "[PATH OF THE EXPERIMENT]"
cfg, model, tokenizer = load_cfg_model_tokenizer(
                          experiment_path,
                          merge=True,
                          device="cpu",
                        )

check_disk_space(model.backbone, "./")

huggingface_hub.login("[Hugging Face API_KEY]")

user_id = huggingface_hub.whoami()["name"]

repo_id = "[The repository name in the Hugging Face platform]"

# push tokenizer to hub
tokenizer.push_to_hub(repo_id=repo_id, private=True)


# push model card to hub
card = get_model_card(cfg, model, repo_id)
card.push_to_hub(
        repo_id=repo_id, repo_type="model", commit_message="Upload model card"
)

# push config to hub
api = huggingface_hub.HfApi()
api.upload_file(
      path_or_fileobj=f"{experiment_path}/cfg.yaml",
      path_in_repo="cfg.yaml",
      repo_id=repo_id,
      repo_type="model",
      commit_message="Upload cfg.yaml",
)

# push model to hub
model.backbone.config.custom_pipelines = {
      "text-generation": {
          "impl": "h2oai_pipeline.H2OTextGenerationPipeline",
          "pt": "AutoModelForCausalLM",
      }
}

model.backbone.push_to_hub(
    repo_id=repo_id,
    private=True,
    commit_message="Upload model",
    safe_serialization=False,
)

# push pipeline to hub
template_env = Environment(
      loader=FileSystemLoader(searchpath="llm_studio/src/")
)

pipeline_template = template_env.get_template("h2oai_pipeline_template.py")

data = {
    "text_prompt_start": cfg.dataset.text_prompt_start,
    "text_answer_separator": cfg.dataset.text_answer_separator,
}

if cfg.dataset.add_eos_token_to_prompt:
    data.update({"end_of_sentence": cfg._tokenizer_eos_token})
else:
    data.update({"end_of_sentence": ""})

custom_pipeline = pipeline_template.render(data)

custom_pipeline_path = os.path.join(experiment_path, "h2oai_pipeline.py")
with open(custom_pipeline_path, "w") as f:
    f.write(custom_pipeline)

api.upload_file(
    path_or_fileobj=custom_pipeline_path,
    path_in_repo="h2oai_pipeline.py",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload h2oai_pipeline.py",
)
```
