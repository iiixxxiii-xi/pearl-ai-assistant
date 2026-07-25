"""MCP Inventory Server — 库存工具 MCP 服务

将 inventory.py 的库存查询功能封装为 MCP (Model Context Protocol) 标准服务。
Agent 通过 MCP Client 发现并调用这些工具，而非直接 import inventory 模块。

启动方式（独立进程）：
    python mcp_inventory_server.py

通信协议：stdio (JSON-RPC)
"""

import sys
import os
import json
import asyncio

# 确保能从项目根目录 import inventory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# ---- 初始化 inventory 模块 ----
import inventory

server = Server("pearl-inventory")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """向 MCP Client 暴露可用工具列表"""
    return [
        types.Tool(
            name="check_inventory",
            description="查询珍珠库存。根据关键词搜索库存中的珠子，返回名称、价格、库存量、品质等。有购买意向时必须调用。支持中文。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，如 '澳白 10mm'、'大溪地'、'淡水珍珠 耳钉'",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="list_all_inventory",
            description="列出所有有货的库存珠子，不筛选。当不确定搜索关键词时使用。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="get_inventory_item",
            description="按名称精确查找某颗珠子的详细信息（价格、库存、品质等）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "珠子全名，如 '澳白珍珠10-11mm'",
                    },
                },
                "required": ["name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """执行工具调用，返回结果"""
    try:
        if name == "check_inventory":
            query = arguments.get("query", "")
            results = inventory.search(query, top_k=20)
            if not results:
                return [types.TextContent(
                    type="text",
                    text="暂无匹配的库存。可以试试其他关键词，或者用 list_all_inventory 查看全部库存。",
                )]
            text = inventory.format_inventory_for_prompt(results)
            return [types.TextContent(type="text", text=text)]

        elif name == "list_all_inventory":
            all_items = [i for i in inventory.list_all() if i.get("stock", 0) > 0]
            if not all_items:
                return [types.TextContent(type="text", text="当前没有有货的库存。")]
            text = inventory.format_inventory_for_prompt(all_items)
            return [types.TextContent(type="text", text=text)]

        elif name == "get_inventory_item":
            item_name = arguments.get("name", "")
            all_items = inventory.list_all()
            for item in all_items:
                if item["name"] == item_name:
                    text = inventory.format_inventory_for_prompt([item])
                    return [types.TextContent(type="text", text=text)]
            return [types.TextContent(
                type="text",
                text=f"未找到名为「{item_name}」的珠子，请检查名称是否正确。",
            )]

        else:
            return [types.TextContent(
                type="text",
                text=f"未知工具：{name}",
            )]

    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"工具执行出错：{e}",
        )]


async def main():
    """启动 MCP stdio 服务器"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
