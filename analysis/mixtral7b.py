# The mistralai/mixtral-8x7b-instruct-v0.1 model can stream output as it's running.
import replicate
import os

os.environ["REPLICATE_API_TOKEN"] = "r8_VhkQUk5QGQK2DdK8CoGgoTSe2cepOHT47F2TZ"
data = """
1.2, 0.10
1.3, 0.11
1.4, 0.12
1.5, 0.13
1.7, 0.15
1.8, 0.16
1.9, 0.18
2.0, 0.19
2.1, 0.21
2.2, 0.23
2.3, 0.25
2.4, 0.27
2.5, 0.29
2.7, 0.31
2.8, 0.33
2.9, 0.35
3.0, 0.38
3.1, 0.40
3.2, 0.43
3.3, 0.45
3.4, 0.48
3.5, 0.50
3.7, 0.53
3.8, 0.56
3.9, 0.59
4.0, 0.62
4.1, 0.64
4.2, 0.67
4.3, 0.70
4.4, 0.74
4.5, 0.77
4.7, 0.80
4.8, 0.83
4.9, 0.86
5.0, 0.90
5.1, 0.93
5.2, 0.96
5.3, 1.0
5.4, 1.0
5.5, 1.1
5.7, 1.1
5.8, 1.1
5.9, 1.2
6.0, 1.2
6.1, 1.3
6.2, 1.3
6.3, 1.3
6.4, 1.4
6.6, 1.4
6.7, 1.4
6.8, 1.5
6.9, 1.5
7.0, 1.6
7.1, 1.6
"""
prompt = "Analyze the trends given in the data provided. Give descriptions of the relationships you can notice within the data : " + data
print(prompt)
iterator = replicate.run(
  "mistralai/mistral-7b-instruct-v0.2",
  input={
        "top_k": 50,
        "top_p": 0.9,
        "prompt": prompt,
        "temperature": 0.6,
        "max_new_tokens": 1024,
        "prompt_template": "<s>[INST] {prompt} [/INST] ",
        "presence_penalty": 0,
        "frequency_penalty": 0
    },
)
joined_string = "".join(iterator)
print(joined_string)