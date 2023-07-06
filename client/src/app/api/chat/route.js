import { Configuration, OpenAIApi } from 'openai-edge'
import { OpenAIStream, StreamingTextResponse } from 'ai'

const config = new Configuration({
  apiKey: process.env.OPENAI_API_KEY
})
const openai = new OpenAIApi(config)

export const runtime = 'edge'

export async function POST(req) {
  const { messages } = await req.json()

  const context = await fetch(`http://127.0.0.1:8000/context?search=${messages[messages.length - 1].content}`)
  const result = await context.json();

  console.log(result)

  const message_prompt = "Act as a Java tutor who repeatedly gives hints instead of direct answers and celebrates when you get it 100% right. You should not give answers if user asks question first, but try to give hints to user instead. Instead of answering questions you should give hints first."
  const ASSISTANT = {"role": "system", "content": message_prompt + "\n" +  "Use this string as context from the book: " + result.context.join("\r\n")};


  const response = await openai.createChatCompletion({
    model: 'gpt-3.5-turbo',
    stream: true,
    messages: [ASSISTANT, ...messages.map((message) => ({
      content: message.content,
      role: message.role
    }))]
  })

  const stream = OpenAIStream(response)
  return new StreamingTextResponse(stream)
}