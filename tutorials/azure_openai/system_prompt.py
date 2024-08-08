import azure_v2 as az2


def generate_system_message(prompt):

    api_key, api_version, endpoint, deployment = az2.load_env()

    # create the azure AzureOpenAI interface in order to connect to the model deployment
    chat = az2.AzureInterface(api_key, api_version, endpoint, deployment)

    response = chat.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. A good format to follow is `Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO.`.\n\nMake it as concise as possible. Include nothing but the system prompt in your response.\n\nFor example, never write: `\"$SYSTEM_PROMPT_HERE\"`.\n\nIt should be like: `$SYSTEM_PROMPT_HERE`."
            },
            {
                "role": "user",
                "content": prompt.strip(),
            }
        ],
    )

    return response


if __name__ == "__main__":
    prompt = "A model that takes in a cybersecurity-oriented question, and responds with a well-reasoned, very short and concise answer. The model is also able to solve coding tasks that are specific to cybersecurity. It is aware of the latest tools that are used in the domain, such as nmap, wfuzz, and many more. Responses related to coding and command specific tasks must only contain the actual code or the actual command to be used."

    system_message = generate_system_message(prompt)
    print(system_message)
