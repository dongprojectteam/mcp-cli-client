// mcp-orchestrator-client.ts
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

import { config } from "dotenv";
import { createInterface } from "readline/promises";
import { OpenAI } from "openai";

import { LLM } from "./src/llm/llm";
import { Claude } from "./src/llm/anthropic"
import { GPT } from "./src/llm/gpt"

config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY is not set");

type MCPServer = {
  name: string;
  client: Client;
  transport: StdioClientTransport;
  functions: OpenAI.ChatCompletionTool[];
};

class MCPOrchestrator {
  // private llm: OpenAI;
  private llm: LLM
  private systemPrompt: string;
  private servers = new Map<string, MCPServer>();
  private allFunctions: OpenAI.ChatCompletionTool[] = [];

  constructor(llm: LLM, systemPrompt: string) {
    this.llm = llm;
    this.systemPrompt = systemPrompt;
  }

  async registerServer(name: string, scriptPath: string) {
    const command = scriptPath.endsWith(".py")
      ? process.platform === "win32" ? "python" : "python3"
      : process.execPath;

    const transport = new StdioClientTransport({ command, args: [scriptPath] });
    const client = new Client({ name: `mcp-${name}`, version: "1.0.0" });
    await client.connect(transport);

    const toolsResult = await client.listTools();
    const functions: OpenAI.ChatCompletionTool[] = toolsResult.tools.map((tool): OpenAI.ChatCompletionTool => ({
      type: "function",
      function: {
        name: `${name}_${tool.name}`.replace(/[^a-zA-Z0-9_-]/g, "_"),
        description: `[${name}] ${tool.description}`,
        parameters: tool.inputSchema,
      },
    }));

    this.servers.set(name, { name, client, transport, functions });
    this.allFunctions.push(...functions);

    console.log(`\u2705 Registered MCP Server '${name}' with tools: ${toolsResult.tools.map(t => t.name).join(", ")}`);
  }

  async processQuery(query: string): Promise<string> {
    const messages: OpenAI.ChatCompletionMessageParam[] = [
      { role: "system", content: this.systemPrompt },
      { role: "user", content: query },
    ];

    const result = await this.llm.chat(messages, this.allFunctions, "auto");
    const response = result.choices[0].message;
    if (!response.tool_calls?.length) return response.content ?? "";

    for (const toolCall of response.tool_calls) {
      const toolMeta = this.allFunctions.find(f => f.function.name === toolCall.function.name);
      if (!toolMeta) throw new Error(`Unknown tool name: ${toolCall.function.name}`);

      const serverName = toolMeta.function.name.split("_")[0];
      const server = this.servers.get(serverName);
      if (!server) throw new Error(`Unknown server: ${serverName}`);

      const toolName = toolMeta.function.name.replace(`${serverName}_`, "");
      const args = JSON.parse(toolCall.function.arguments);

      const toolResult = await server.client.callTool({ name: toolName, arguments: args });

      const content = typeof toolResult.content === "string"
        ? toolResult.content
        : JSON.stringify(toolResult.content ?? "");

      messages.push(
        { role: "assistant", tool_calls: [toolCall] },
        { role: "tool", tool_call_id: toolCall.id, content }
      );
    }

    const final = await this.llm.chat(messages);
    return final.choices[0].message?.content ?? "";
  }

  async cleanup() {
    for (const server of this.servers.values()) {
      await server.client.close();
    }
  }
}

import fs from "fs/promises";

async function registerServersFromConfig(orchestrator: MCPOrchestrator, configPath: string) {
  const content = await fs.readFile(configPath, "utf-8");
  const config: Record<string, { command: string; args: string[]; env?: Record<string, string> }> = JSON.parse(content);

  for (const [name, { command, args, env }] of Object.entries(config)) {
    const transport = new StdioClientTransport({ command, args, env });
    const client = new Client({ name: `mcp-${name}`, version: "1.0.0" });
    await client.connect(transport);

    const toolsResult = await client.listTools();
    const functions = toolsResult.tools.map((tool): OpenAI.ChatCompletionTool => ({
      type: "function",
      function: {
        name: `${name}_${tool.name}`.replace(/[^a-zA-Z0-9_-]/g, "_"),
        description: `[${name}] ${tool.description}`,
        parameters: tool.inputSchema,
      },
    }));

    orchestrator["servers"].set(name, { name, client, transport, functions });
    orchestrator["allFunctions"].push(...functions);

    console.log(`✅ Registered MCP server '${name}' from config with tools: ${toolsResult.tools.map(t => t.name).join(", ")}`);
  }
}

async function loadConfig(): Promise<{ llm: LLM; systemPrompt: string, serverConfig: string }> {
  const configPath = "./mcp.llm.json";
  const content = await fs.readFile(configPath, "utf-8");
  const { llm: model, systemPrompt, serverConfig } = JSON.parse(content);

  let llm: LLM;
  if (model === "claude") {
    const CLAUDE_API_KEY = process.env.CLAUDE_API_KEY;
    if (!CLAUDE_API_KEY) throw new Error("CLAUDE_API_KEY is not set");
    llm = new Claude(CLAUDE_API_KEY);
  } else if (model === "gpt") {
    const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
    if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY is not set");
    llm = new GPT(OPENAI_API_KEY);
  } else {
    throw new Error(`Unsupported LLM type: '${model}'. Use 'gpt' or 'claude' in mcp.llm.json`);
  }

  if (typeof systemPrompt !== "string" || !systemPrompt.trim()) {
    throw new Error("Missing or invalid 'systemPrompt' in mcp.llm.json");
  }

  if (typeof serverConfig !== "string" || !serverConfig.trim()) {
    throw new Error("Missing or invalid 'serverConfig' in mcp.llm.json");
  }

  return { llm, systemPrompt, serverConfig };
}

async function main() {
  const { llm, systemPrompt, serverConfig } = await loadConfig();
  const orchestrator = new MCPOrchestrator(llm, systemPrompt);
  await registerServersFromConfig(orchestrator, serverConfig);

  const rl = createInterface({ input: process.stdin, output: process.stdout });

  try {
    console.log("\nMCP Orchestrator Client Started! (LLM loaded from mcp.llm.json)");
    console.log("Type your queries or 'quit' to exit.");

    while (true) {
      const message = await rl.question("\nQuery: ");
      if (message.toLowerCase() === "quit") break;
      const response = await orchestrator.processQuery(message);
      console.log("\n" + response);
    }
  } finally {
    rl.close();
    await orchestrator.cleanup();
    process.exit(0);
  }
}

main()
