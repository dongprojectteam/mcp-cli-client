import { OpenAI } from "openai";
import { LLM } from "./llm";

export class GPT implements LLM {
  private openai: OpenAI;

  constructor(apiKey: string) {
    this.openai = new OpenAI({ apiKey });
  }

  async chat(messages: any[], tools: any[] = [], tool_choice: "auto" | "none" = "auto") {
    const res = await this.openai.chat.completions.create({
      model: "gpt-4-1106-preview",
      messages,
      tools,
      tool_choice,
    });
    return res;
  }
}
