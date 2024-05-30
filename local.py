from llama_cpp import Llama 
import timeit

llm = Llama(model_path="llama-2-7b-chat.ggmlv3.q2_K.bin",
            n_ctx=512,
            n_batch=128)

start = timeit.default_timer()

prompt = "What is Python?"

output = llm(prompt,
    max_tokens=-1,
    echo=False,
    temperature=0.1,
    top_p=0.9)

stop = timeit.default_timer()
duration = stop - start
print("Time: ", duration, '\n\n')

print(output['choices'][0]['text'])

with open("response.txt", "a") as f:
f.write(f"Time: {duration}")
f.write(output['choices'][0]['text'])