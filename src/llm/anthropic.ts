import Anthropic from "@anthropic-ai/sdk";
import { LLM } from "./llm";
function isTextBlock(block: any): block is { type: "text"; text: string } {
    return block.type === "text" && typeof block.text === "string";
}

export class Claude implements LLM {
    private anthropic: Anthropic;

    constructor(apiKey: string) {
        this.anthropic = new Anthropic({ apiKey });
    }

    async chat(messages: any[], tools: any[] = [], tool_choice: "auto" | "none" = "auto", max_tokens = 1024) {
        const res = await this.anthropic.messages.create({
            model: "claude-3-opus-20240229",
            messages,
            system: messages.find(m => m.role === "system")?.content,
            tools,
            max_tokens: 1024,
        });

        const firstTextBlock = res.content.find(isTextBlock) as { type: "text"; text: string } | undefined;

        return {
            choices: [{
                message: {
                    content: firstTextBlock?.text ?? "",
                    tool_calls: [],
                },
            }],
        };
    }
}