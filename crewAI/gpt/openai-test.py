from openai import OpenAI

import configparser


config = configparser.ConfigParser()
config.read('credentials.ini')

client = OpenAI(
    api_key = config['OpenAI']['api_key']
)

response = client.chat.completions.create(
  #model="gpt-3.5-turbo",
  model="gpt-4o",
  messages=[
    {
      "role": "user",
      "content": "help me naming a dog"
    }
  ],
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response.choices[0].message.content)