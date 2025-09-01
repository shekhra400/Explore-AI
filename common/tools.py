from langchain_community.agent_toolkits import SlackToolkit


async def get_slack_tool():
    """
    Initializes a ToolNode using a Slack-aware MCP tool from a remote MCP server.

    This function:
    - Connects to a remote MCP server using MultiServerMCPClient.
    - Retrieves available tools from the 'slack' server context.
    - Wraps the first available tool in a LangGraph ToolNode for use in a StateGraph.

    Returns:
        ToolNode: A LangGraph-compatible node that can be added to your graph.
    """

    toolkit = SlackToolkit()

    tools = toolkit.get_tools()
    return tools

