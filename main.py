"""
Query Decomposition Agent - Unified Component
============================================

This is the main entry point that:
1. Identifies intention (Analytics, Action, Warnings, etc.)
2. Checks if clarification/specification is needed
3. Determines data requirements if no HITL needed

Uses SINGLE LLM call with both tool calling and structured output.
"""

import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage
from harri_logging import SimpleLogger
from harri_gen_ai_agent.engines.lc import HarriBaseAgent, fetch_prompt, format_chat_messages
from harri_gen_ai_agent.engines.state import QueryDecompositionAnalysis, ClarificationType
from harri_gen_ai_agent.engines.lg.tools.generic_tools import _search_employees_impl, search_employees

_logger = SimpleLogger(component="intention_identifier")

class QueryDecompositionResponse(BaseModel):
    """
    Return this analysis after processing the user's query.
    Use this tool to provide your final analysis of the user's intention.
    """
    
    # Core intention identification
    intention: str = Field(description="Primary intention: ANALYTICS, PERFORM_ACTION, REQUEST_WARNINGS_LIST, REQUEST_FIX_OPTIONS, or CONVERSATIONAL")
    confidence: float = Field(description="Confidence level 0.0-1.0 for intention classification")
    
    # HITL assessment
    needs_clarification: bool = Field(description="Whether clarification is needed due to ambiguity")
    needs_specification: bool = Field(description="Whether specification is needed due to broad scope")
    clarification_type: Optional[str] = Field(default=None, description="CLARIFICATION or SPECIFICATION")
    clarification_message: Optional[str] = Field(default=None, description="Message to ask user for clarification/specification")
    # clarification_options: List[str] = Field(description="Specific options to present to user")

    # Data requirements (for context optimization when no HITL)
    needs_context_optimization: bool = Field(description="Whether context optimization is needed")
    should_fetch_employees: bool = Field(description="Whether employee data is needed")
    should_fetch_schedule: bool = Field(description="Whether schedule data is needed")
    should_fetch_reports: bool = Field(description="Whether analytics reports are needed")
    should_fetch_warnings_list: bool = Field(description="Whether warnings data is needed")
    should_fetch_knowledge_base: bool = Field(description="Whether knowledge base docs are needed")


INTENTIONS = [
    ["REQUEST_WARNINGS_LIST", "When the user wants a list of the schedule warnings or compliance issues"],
    ["REQUEST_FIX_OPTIONS", "When the user wants to know how to fix a warning "
                        "but not performing the actual fix action"],
    ["PERFORM_ACTION", "When the user wants to perform an action, for example perform a fix action, reassigning, unassign, etc.."],
    ["ANALYTICS", "When the user wants to know about analytics and insights, for example, labor report, labor anomalies, etc.."],
    ["CONVERSATIONAL", "When the user is interacting with the Agent for general schedule purposes/intentions "
                "including but not limited to, summarization of schedule/schedule warnings, more details on a warning or list of warnings, questions about persons,"
                " employees, entities, attendance-related queries, and general conversational queries, How-to, Guide me, etc.."
                " discussions on specific cases, listing details, etc.."],
]

AGENT_MAPPING = {
    "REQUEST_WARNINGS_LIST": "run_warnings_list",
    "REQUEST_FIX_OPTIONS": "run_fix_options", 
    "PERFORM_ACTION": "run_perform_action",
    "ANALYTICS": "run_analytics_agent",
    "CONVERSATIONAL": "run_default_agent"
}


class QueryDecompositionAgent(HarriBaseAgent):
    """
    Unified Query Decomposition Agent
    
    This is the main component that:
    1. Identifies user intention
    2. Checks if clarification/specification is needed (HITL)
    3. Determines data requirements for context optimization
    """
    
    def __init__(self, llm, memory):
        super().__init__(llm, memory)
        # Store the original LLM for unified tool calling + structured output
        self.llm = llm
        
    def _generate_lightweight_context(self, session_context: Dict) -> Dict:
        """Generate lightweight context summary for LLM analysis"""
        
        lightweight = {
            "employees_summary": self._summarize_employees(session_context.get("schedule_context", {}).get("employees", [])),
            "schedule_summary": self._summarize_schedule(session_context.get("schedule_context", {}).get("schedule", [])),
            "reports_summary": self._summarize_reports(session_context.get("reports", {})),
            "warnings_summary": self._summarize_warnings(session_context.get("schedule_context", {}).get("warnings_obj", {}))
        }
        
        return lightweight

    def _summarize_employees(self, employees: List[Dict]) -> Dict:
        """Create lightweight employee summary"""
        if not employees:
            return {"count": 0, "sample_names": [], "sample_roles": []}
        
        names = []
        roles = set()
        
        for emp in employees[:50]:  # Sample first 50 employees
            # Try to build full name
            first_name = emp.get("first_name", "")
            last_name = emp.get("last_name", "")
            name = emp.get("name", "")
            
            if first_name and last_name:
                full_name = f"{first_name} {last_name}"
            elif name:
                full_name = name
            else:
                full_name = f"Employee {emp.get('id', 'Unknown')}"
            
            names.append(full_name)
            
            # Extract roles
            positions = emp.get("positions", [])
            if positions:
                for pos in positions:
                    if isinstance(pos, dict) and pos.get("role"):
                        roles.add(pos["role"])
            elif emp.get("role"):
                roles.add(emp["role"])
        
        return {
            "count": len(employees),
            "sample_names": names[:10],  # First 10 names
            "sample_roles": list(roles)[:10],  # Up to 10 unique roles
            "has_more": len(employees) > 50
        }

    def _summarize_schedule(self, schedule: List[Dict]) -> Dict:
        """Create lightweight schedule summary"""
        if not schedule:
            return {"count": 0, "date_ranges": [], "shift_types": []}

        assigned_shifts = 0
        unassigned_shifts = 0

        first_schedule = schedule[0]

        for role in first_schedule["roles"]:
            for role_day in role["role_days"]:
                for assignee in role_day["assignees"]:
                    if assignee["type"] != "VIRTUAL":
                        assigned_shifts += len(assignee["assignee_shifts"])
                    else:
                        unassigned_shifts += len(assignee["assignee_shifts"])

        shifts_count = assigned_shifts + unassigned_shifts

        return {
            "shifts_count": shifts_count,
            "unassigned_shifts_count": unassigned_shifts,
            "assigned_shifts_count": assigned_shifts,
            "has_more": shifts_count > 100
        }

    def _summarize_reports(self, reports: Dict) -> Dict:
        """Create dynamic lightweight reports summary"""
        if not reports:
            return {"available": False, "count": 0, "report_keys": []}
        
        report_keys = []
        date_ranges = []
        data_categories = set()
        
        for report_key, report_data in reports.items():
            report_keys.append(report_key)
            
            # Recursively extract date ranges and data categories
            self._extract_report_info(report_data, date_ranges, data_categories)
        
        return {
            "available": True,
            "count": len(reports),
            "report_keys": report_keys[:5],
            "date_ranges": date_ranges[:5],
            "data_categories": list(data_categories)[:10],
            "has_more_reports": len(reports) > 5
        }

    def _extract_report_info(self, data, date_ranges, data_categories, depth=0):
        """Recursively extract report information"""
        if depth > 4 or not isinstance(data, dict):
            return
            
        for key, value in data.items():
            # Look for date patterns
            if isinstance(value, str) and self._looks_like_date(value):
                if value not in date_ranges:
                    date_ranges.append(value)
            elif key in ["from_date", "to_date", "date", "start_date", "end_date"] and isinstance(value, str):
                if value not in date_ranges:
                    date_ranges.append(value)
            
            # Identify data categories
            if isinstance(value, dict) and len(value) > 0:
                data_categories.add(key)
                self._extract_report_info(value, date_ranges, data_categories, depth + 1)

    def _looks_like_date(self, value: str) -> bool:
        """Check if a string looks like a date"""
        if not isinstance(value, str):
            return False
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY  
            r'\w{3} \d{2}, \d{4}'  # Mon DD, YYYY
        ]
        return any(re.match(pattern, value) for pattern in date_patterns)

    def _summarize_warnings(self, warnings_obj: Dict) -> Dict:
        """Create dynamic lightweight warnings summary"""
        if not warnings_obj:
            return {"available": False, "count": 0, "warning_codes": []}
        
        warnings_list = self._find_warnings_list(warnings_obj)
        if not warnings_list:
            return {"available": False, "count": 0, "warning_codes": []}
        
        warning_codes = set()
        affected_entities = {}
        
        for warning in warnings_list[:30]:  # Sample first 30 warnings
            if not isinstance(warning, dict):
                continue
                
            # Extract warning codes
            if warning.get("code"):
                warning_codes.add(warning.get("code"))
            
            # Track affected entities
            for field_name, field_value in warning.items():
                if isinstance(field_value, dict) and field_value.get("id"):
                    if field_name not in affected_entities:
                        affected_entities[field_name] = set()
                    affected_entities[field_name].add(field_value.get("id"))
        
        affected_entity_counts = {
            entity_type: len(ids) for entity_type, ids in affected_entities.items()
        }
        
        return {
            "available": True,
            "count": len(warnings_list),
            "warning_codes": list(warning_codes)[:15],
            "affected_entity_counts": affected_entity_counts,
            "has_more": len(warnings_list) > 30
        }

    def _find_warnings_list(self, warnings_obj: Dict) -> list:
        """Dynamically find the warnings list in the object"""
        if isinstance(warnings_obj, list):
            return warnings_obj
        elif isinstance(warnings_obj, dict):
            for key in ["warnings", "warning_list", "items", "data", "results"]:
                if key in warnings_obj and isinstance(warnings_obj[key], list):
                    return warnings_obj[key]
            
            for value in warnings_obj.values():
                if isinstance(value, list) and len(value) > 0:
                    first_item = value[0]
                    if isinstance(first_item, dict):
                        warning_indicators = ["message", "code", "type", "severity", "error", "issue"]
                        if any(indicator in first_item for indicator in warning_indicators):
                            return value
        
        return []
    
    def _format_employees_summary(self, emp: Dict) -> str:
        """Format employees summary for prompt"""
        if emp.get("count", 0) > 0:
            return f"{emp.get('count', 0)} employees available\n"
        return "No employees loaded"

    def _format_schedule_summary(self, sch: Dict) -> str:
        """Format schedule summary for prompt"""
        if sch.get("shifts_count", 0) > 0:
            return (
                f"{sch.get('assigned_shifts_count', 0)} assigned shifts available\n"
                f"{sch.get('unassigned_shifts_count', 0)} unassigned shifts available\n"
            )
        return "No schedule loaded"

    def _format_warnings_summary(self, wrn: Dict) -> str:
        """Format warnings summary for prompt"""
        if wrn.get("available", False):
            return (
                f"{wrn.get('count', 0)} warnings found\n"
                f"Warning codes: {', '.join(wrn.get('warning_codes', []) or [])}"
            )
        return "Not available"
    
    async def ainvoke(self, query, chain_type=None) -> QueryDecompositionAnalysis:
        """
        Main entry point for unified query decomposition using SINGLE LLM call.
        
        TURN-BASED TOOL CALLING ARCHITECTURE:
        =====================================
        The LLM operates in a turn-based loop (not transactional):
        
        1. LLM decides which tool to call (search_employees or respond_query_decomposition)
        2. Tool executes in Python runtime
        3. Tool result is sent back to the LLM as a ToolMessage
        4. LLM reasons over the result
        5. LLM decides the next tool to call
        
        This loop continues until the LLM calls respond_query_decomposition (the response tool).
        
        Available Tools:
        ---------------
        1. search_employees - For employee name resolution (optional)
        2. respond_query_decomposition - For structured output (REQUIRED as final call)
        
        Example Flow:
        ------------
        Query: "Assign John to the 8-3 shift"
        
        Turn 1: LLM → search_employees(name_query="John")
        Turn 2: Runtime → Returns {"matches": [{"id": 42, ...}], ...}
        Turn 3: LLM reasons → John is ID 42, not ambiguous
        Turn 4: LLM → respond_query_decomposition(..., matched_employee_ids=[42])
        Turn 5: Runtime → Returns QueryDecompositionAnalysis ✓
        
        Args:
            query: {
                "message": str,
                "session_context": dict,
                "current_view": dict,
                "brand_id": str
            }
        
        Returns:
            QueryDecompositionAnalysis with intention and HITL decision
        """
        # Generate lightweight context for data-aware analysis
        lightweight_context = self._generate_lightweight_context(query.get("session_context", {}))
        
        # Format summaries using helper methods
        employees_summary_text = self._format_employees_summary(lightweight_context["employees_summary"])
        schedule_summary_text = self._format_schedule_summary(lightweight_context["schedule_summary"])
        warnings_summary_text = self._format_warnings_summary(lightweight_context["warnings_summary"])
        
        # FETCH PROMPT TEMPLATE FOR INTENTION IDENTIFICATION
        query_decomposition_template = fetch_prompt("query-decomposition-prompt", commit="latest")
        template_text = query_decomposition_template.prompt
        
        # Create unified prompt that handles both employee resolution AND structured analysis
        unified_prompt = PromptTemplate(
            template=template_text,
            input_variables=["input", "history", "current_view"],
            partial_variables={
                "intentions": "\n- ".join([": ".join(intention) for intention in INTENTIONS]),
                "current_time": datetime.datetime.now().strftime("%A, %Y-%m-%dT%H:%M:%S"),
                "employees_summary_text": employees_summary_text,
                "schedule_summary_text": schedule_summary_text,
                "warnings_summary_text": warnings_summary_text,
            },
            template_format="jinja2",
        )

        input_dict = {
            "input": query.get("message", ""),
            "history": format_chat_messages(self.memory),
            "current_view": query.get("current_view", {})
        }
        
    
        # Import respond_query_decomposition tool
        from harri_gen_ai_agent.engines.lg.tools.generic_tools import respond_query_decomposition
        
        # Bind BOTH tools: search_employees + respond_query_decomposition
        llm_with_tools = self.llm.bind_tools(
            [search_employees, respond_query_decomposition],
            parallel_tool_calls=True
        )
        
        # Prepare message with full context
        formatted_prompt = unified_prompt.format(**input_dict)
        messages = [HumanMessage(content=formatted_prompt)]
        
        _logger.info("Invoking unified LLM call with search_employees and respond_query_decomposition tools")
        
        # Store schedule context for tool injection
        schedule_context = query.get("session_context", {}).get("schedule_context", {})
        
        # ==============================================================
        # TURN-BASED AGENTIC LOOP (Mental Model)
        # ==============================================================
        # 1. LLM decides which tool to call
        # 2. Tool executes in runtime
        # 3. Tool result sent back to LLM
        # 4. LLM reasons over result
        # 5. LLM decides next tool to call
        # Loop continues until respond_query_decomposition is called
        # ==============================================================
        final_analysis = None
        max_iterations = 2
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # ==============================================================
            # TURN 1: LLM decides which tool to call
            # ==============================================================
            _logger.info(f"Iteration {iteration}: LLM deciding which tool to call...")
            
            # Invoke LLM
            ai_message = await llm_with_tools.ainvoke(messages)
            
            # Check for tool calls
            tool_calls = getattr(ai_message, "tool_calls", []) or []
            _logger.info(f"Iteration {iteration}: LLM called {len(tool_calls)} tool(s)")
            
            if not tool_calls:
                _logger.warning("No tool calls received from LLM")
                break
            
            # ==============================================================
            # TURN 2: Tools execute in runtime
            # ==============================================================
            tool_messages = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")
                
                _logger.info(f"Executing tool: {tool_name}")
                
                if tool_name == "search_employees":
                    # Execute search_employees with injected schedule_context
                    result = _search_employees_impl(
                        schedule_context=schedule_context,
                        name_query=tool_args.get("name_query"),
                        role_hint=tool_args.get("role_hint"),
                        team_hint=tool_args.get("team_hint"),
                        is_generic_query=tool_args.get("is_generic_query", False)
                    )
                    
                    # ==============================================================
                    # TURN 3: Tool result sent back to LLM
                    # ==============================================================
                    _logger.info(f"Tool result: {len(result.get('matches', []))} matches found")
                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call_id
                        )
                    )
                    
                elif tool_name == "respond_query_decomposition":
                    # This is the final structured output - end of loop
                    _logger.info("LLM called respond_query_decomposition - finalizing analysis")
                    
                    # Extract matched_employee_ids and process clarifications
                    matched_employee_ids = tool_args.get("matched_employee_ids", [])
                    
                    # Check if we need to generate clarification messages for employees
                    clarification_message = tool_args.get("clarification_message", "")
                    
                    # Convert to QueryDecompositionAnalysis
                    final_analysis = QueryDecompositionAnalysis(
                        intention=tool_args.get("intention"),
                        target_agent=tool_args.get("target_agent"),
                        confidence=tool_args.get("confidence", 0.0),
                        needs_clarification=tool_args.get("needs_clarification", False),
                        needs_specification=tool_args.get("needs_specification", False),
                        clarification_type=ClarificationType(tool_args.get("clarification_type")) if tool_args.get("clarification_type") else None,
                        clarification_message=clarification_message,
                        needs_context_optimization=tool_args.get("needs_context_optimization", False),
                        # should_fetch_employees=False,
                        should_fetch_schedule=tool_args.get("should_fetch_schedule", False),
                        should_fetch_reports=tool_args.get("should_fetch_reports", False),
                        should_fetch_warnings_list=tool_args.get("should_fetch_warnings_list", False),
                        should_fetch_knowledge_base=tool_args.get("should_fetch_knowledge_base", False),
                        matched_employee_ids=matched_employee_ids,
                    )
                    
                    _logger.info(f"Final analysis created: intention={final_analysis.intention}, needs_clarification={final_analysis.needs_clarification}")
                    return final_analysis
            
            # ==============================================================
            # TURN 4: Add tool results to conversation for LLM to reason over
            # ==============================================================
            # The LLM will receive these results and decide the next action
            if tool_messages:
                messages.append(ai_message)  # Add LLM's tool calls
                messages.extend(tool_messages)  # Add tool results
                _logger.info(f"Added {len(tool_messages)} tool result(s) to conversation - LLM will reason over them next")
        
        # Fallback if max iterations reached without respond_query_decomposition
        _logger.error("Max iterations reached without respond_query_decomposition call")
        
        # Return a default CONVERSATIONAL analysis as fallback
        _logger.warning("Falling back to default CONVERSATIONAL analysis")
        return QueryDecompositionAnalysis(
            intention="CONVERSATIONAL",
            target_agent="run_default_agent",
            confidence=0.5,
            needs_clarification=False,
            needs_specification=False,
            clarification_type=None,
            clarification_message="",
            needs_context_optimization=False,
            should_fetch_employees=False,
            should_fetch_schedule=False,
            should_fetch_reports=False,
            should_fetch_warnings_list=False,
            should_fetch_knowledge_base=False,
            matched_employee_ids=[],
        )
        



