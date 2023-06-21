import logging

import click
import torch
from auto_gptq import AutoGPTQForCausalLM

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

from googletrans import Translator

translator = Translator()

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY


def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """

    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        # The code supports all huggingface models that ends with GPTQ and have some variation
        # of .no-act.order or .safetensors in their HF repo.
        logging.info("Using AutoGPTQForCausalLM for quantized models")

        if ".safetensors" in model_basename:
            # Remove the ".safetensors" ending if present
            model_basename = model_basename.replace(".safetensors", "")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        logging.info("Tokenizer loaded")

        model = AutoGPTQForCausalLM.from_quantized(
            model_id,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device="cuda:0",
            use_triton=False,
            quantize_config=None,
        )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
def main(device_type, show_sources):
    """
    This function implements the information retrieval task.


    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
    2. Loads the existing vectorestore that was created by inget.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})

    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    # load the LLM for generating Natural Language responses

    # for HF models
    # model_id = "TheBloke/vicuna-7B-1.1-HF"
    # model_id = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
    # model_id = "TheBloke/guanaco-7B-HF"
    # model_id = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
    # alongside will 100% create OOM on 24GB cards.
    # llm = load_model(device_type, model_id=model_id)

    # for GPTQ (quantized) models
    # model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
    # model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
    # model_id = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
    # model_basename = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors" # Requires
    # ~21GB VRAM. Using STransformers alongside can potentially create OOM on 24GB cards.
    # model_id = "TheBloke/wizardLM-7B-GPTQ"
    # model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"
    model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
    model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
    # model_id = "anon8231489123/vicuna-13b-GPTQ-4bit-128g"
    # model_basename = "vicuna-13b-GPTQ-4bit-128g.compat.no-act.order.safetensors"

    llm = load_model(device_type, model_id=model_id, model_basename=model_basename)

    # Customize the prompt template

    # create our examples
    examples = [
        {
            "query": "How to connect the tablet to the well's Wi-Fi?",
            "answer": "To connect the tablet to the well's Wi-Fi, follow these steps: 1. Turn on the tablet and go to the Wi-Fi settings. 2. Select the well's Wi-Fi network from the list of available networks. 3. Enter the network password when prompted. 4. Wait until the tablet is connected to the well's Wi-Fi."
        },
        {
            "query": "What to do if the overflow is not full?",
            "answer": "If the overflow is not full, you can follow these steps: 1. Check if water is flowing through the line. 2. Increase the opening of the water inlet valve in the overflow. 3. Increase the opening of the needle valve in the line."
        },
        {
            "query": "How to check if water is flowing in the hoses that carry the overflow sample to the colorimeter?",
            "answer": "To check if water is flowing in the hoses, follow these steps: 1. Verify if the hoses are properly connected. 2. Disconnect the hoses to allow air to escape. 3. Check for obstructions in the hoses, such as dust residues, and remove them."
        },
        {
            "query": "What to do if the Fluosilicic Acid and Hypochlorite of Sodium tanks are not full?",
            "answer": "If the Fluosilicic Acid and Hypochlorite of Sodium tanks are not full, you should inform the responsible personnel at CORSAN to fill the tanks."
        }
    ]

    # create the template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """
            Augen is a pioneering company focused on digitalizing water treatment processes. 
            Our advanced systems, Poço 4.0 and ETA 4.0, analyze, treat, and digitally manage water. 
            With modular hardware and software, managers gain access to crucial information anytime, anywhere. 
            Our mission is to provide treated water efficiently while enabling intelligent resource management.
            Key Features of Saneamento 4.0:

                Data transformation into actionable knowledge.
                Plug-and-play or customized solutions.
                Water analysis, treatment, and digital management.
                Seamless integration with diverse technologies.
                Remote operability assistance.
                Configurable and scalable.

            Below are selected excerpts from interactions with Auga, an AI assistant of Augen , a leading technology company. 
            Auga is designed to be cordial, polite, and knowledgeable, offering valuable and educational answers to user inquiries. 
            The following examples showcase Auga's capabilities:
    """

    # and the suffix our user input and output indicator
    suffix = """
    User: {query}
    AI: """

    example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=2048  # this sets the max length that examples should be
    )

    # now create the few shot prompt template
    dynamic_prompt_template  = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
    )
    
    # print(dynamic_prompt_template.format(query="How to connect the tablet to the well's Wi-Fi?"))

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # qa = dynamic_prompt_template.format(qa=qa)
    while True:

        query = input("\nEnter a query: ")
        query = dynamic_prompt_template.format(query=query)
        query = translator.translate(query, src='pt', dest='en').text

        # Aplly the template to the query

        #query = dynamic_prompt_template.format(query=query)

        if query == "exit":
            break

        # Get the answer from the chain

        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        answer = translator.translate(answer, src='en', dest='pt').text

        
        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
