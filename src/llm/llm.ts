export interface LLM {
    chat(
      messages: any[],
      tools?: any[],
      tool_choice?: "auto" | "none"
    ): Promise<any>;
  }
  