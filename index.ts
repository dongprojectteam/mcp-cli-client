// mcp-orchestrator-client.ts
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

import { config } from "dotenv";
import { createInterface } from "readline/promises";
import { OpenAI } from "openai";

config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) throw new Error("OPENAI_API_KEY is not set");

type MCPServer = {
  name: string;
  client: Client;
  transport: StdioClientTransport;
  functions: OpenAI.ChatCompletionTool[];
};

const SYSTEM_PROMPT = "You are Bixby, a helpful and knowledgeable AI agent that coordinates multiple MCP servers. If someone asks who you are, say: 'I am Bixby agent.'";


class MCPOrchestrator {
  private llm: OpenAI;
  private servers = new Map<string, MCPServer>();
  private allFunctions: OpenAI.ChatCompletionTool[] = [];

  constructor(apiKey: string) {
    this.llm = new OpenAI({ apiKey });
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
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: query },
    ];

    const result = await this.llm.chat.completions.create({
      model: "gpt-4-1106-preview",
      messages,
      tools: this.allFunctions,
      tool_choice: "auto",
    });

    const response = result.choices[0].message;
    if (!response.tool_calls?.length) return response.content ?? "";

    for (const toolCall of response.tool_calls) {
      const toolMeta = this.allFunctions.find(f => f.function.name === toolCall.function.name);
      if (!toolMeta) throw new Error(`Unknown tool name: ${toolCall.function.name}`);

      const toolIdParts = toolMeta.function.name.split("_");
      const serverName = toolIdParts[0];
      const server: MCPServer | undefined = this.servers.get(serverName);
      if (!server) throw new Error(`Unknown server: ${serverName}`);

      const toolName = toolMeta.function.name.replace(`${serverName}_`, "");
      const args = JSON.parse(toolCall.function.arguments);

      const toolResult = await server.client.callTool({
        name: toolName,
        arguments: args,
      });

      const content = typeof toolResult.content === "string"
        ? toolResult.content
        : JSON.stringify(toolResult.content ?? "");

      messages.push(
        { role: "assistant", tool_calls: [toolCall] },
        { role: "tool", tool_call_id: toolCall.id, content }
      );
    }

    const final = await this.llm.chat.completions.create({
      model: "gpt-4-1106-preview",
      messages,
    });

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

async function main() {
  const orchestrator = new MCPOrchestrator(OPENAI_API_KEY as string);
  await registerServersFromConfig(orchestrator, "./mcp.servers.json");

  const rl = createInterface({ input: process.stdin, output: process.stdout });

  try {
    console.log("\nMCP Orchestrator Client Started!");
    console.log("Type your queries or 'quit' to exit.");

    while (true) {
      try {
        const message = await rl.question("\nQuery: ");
        if (message.toLowerCase() === "quit") break;

        const response = await orchestrator.processQuery(message);
        console.log("\n" + response);
      } catch (err) {
        console.error("❌ Error:", err);
      }
    }
  } finally {
    rl.close();
    await orchestrator.cleanup();
    process.exit(0);
  }
}

main();
