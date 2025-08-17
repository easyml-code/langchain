"""
Multi-Agent Orchestration Library

A flexible, LLM-agnostic library for orchestrating multiple AI agents with intelligent routing
and dynamic workflow management using LangGraph.

Author: Assistant
Version: 1.0.0
"""

import operator
import json
import uuid
from abc import ABC, abstractmethod
from typing import Annotated, Dict, List, TypedDict, Literal, Any, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
from langgraph.graph import StateGraph, END, START


# Core Types and Structures
class OrchestratorState(TypedDict):
    """State structure for the orchestrator"""
    messages: Annotated[List[Dict], operator.add]
    current_task: str
    agent_status: Dict[str, Dict[str, Any]]  # Track status for each agent
    iteration_count: int
    max_iterations: int
    router_decision: str
    final_response: str
    metadata: Dict[str, Any]


@dataclass
class AgentResponse:
    """Standardized agent response format"""
    content: str
    status: Literal["complete", "in_progress", "failed", "needs_revision"]
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class ExecutionTrace:
    """Trace of execution steps"""
    step_id: str
    timestamp: datetime
    agent_name: str
    action: str
    input_data: str
    output_data: str
    status: str
    duration_ms: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OrchestrationResult:
    """Final result of orchestration"""
    task: str
    final_response: str
    total_iterations: int
    execution_time_ms: int
    agent_status: Dict[str, Dict[str, Any]]
    execution_trace: List[ExecutionTrace]
    metadata: Dict[str, Any] = None
    success: bool = True
    error_message: str = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Abstract Base Classes
class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.execution_count = 0
        
    @abstractmethod
    def execute(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """Execute the agent's main functionality"""
        pass
    
    def can_handle_task(self, task: str, context: Dict[str, Any]) -> float:
        """Return confidence score (0.0-1.0) for handling this task"""
        return 0.5  # Default neutral confidence
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "name": self.name,
            "description": self.description,
            "execution_count": self.execution_count,
            "status": "ready"
        }


class LLMAgent(BaseAgent):
    """Agent that uses any LLM for processing"""
    
    def __init__(self, name: str, description: str, system_prompt: str, llm: Any = None):
        super().__init__(name, description)
        self.system_prompt = system_prompt
        self.llm = llm
    
    def _invoke_llm(self, prompt: str) -> str:
        """Invoke LLM with flexible interface support"""
        if self.llm is None:
            raise ValueError(f"No LLM provided for agent {self.name}")
        
        try:
            # Try different LLM interfaces
            if hasattr(self.llm, 'invoke'):
                # LangChain-style interface
                if hasattr(self.llm, 'bind') or 'langchain' in str(type(self.llm)):
                    from langchain_core.messages import HumanMessage, SystemMessage
                    messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
                    response = self.llm.invoke(messages)
                    return response.content if hasattr(response, 'content') else str(response)
                else:
                    response = self.llm.invoke(prompt)
                    return response.content if hasattr(response, 'content') else str(response)
            
            elif hasattr(self.llm, 'generate') or hasattr(self.llm, '__call__'):
                # Direct callable interface
                full_prompt = f"{self.system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
                response = self.llm(full_prompt)
                return str(response)
            
            elif hasattr(self.llm, 'chat') or hasattr(self.llm, 'complete'):
                # OpenAI-style interface
                if hasattr(self.llm, 'chat'):
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    response = self.llm.chat(messages=messages)
                else:
                    full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                    response = self.llm.complete(prompt=full_prompt)
                
                # Extract content from response
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    return response.choices[0].message.content
                elif hasattr(response, 'content'):
                    return response.content
                else:
                    return str(response)
            
            else:
                # Fallback: treat as string formatter
                full_prompt = f"{self.system_prompt}\n\nTask: {prompt}"
                return str(self.llm).format(prompt=full_prompt)
                
        except Exception as e:
            return f"Error invoking LLM: {str(e)}"
    
    def execute(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """Execute using LLM"""
        self.execution_count += 1
        
        try:
            # Prepare input with context
            input_prompt = f"Task: {task}\n\nContext: {json.dumps(context, indent=2)}"
            
            # Execute LLM
            response_content = self._invoke_llm(input_prompt)
            
            # Parse response for status indicators
            content_lower = response_content.lower()
            
            # Determine status based on content
            if any(phrase in content_lower for phrase in ["complete", "finished", "done", "ready", "completed"]):
                status = "complete"
                confidence = 0.9
            elif any(phrase in content_lower for phrase in ["progress", "working", "developing", "in progress"]):
                status = "in_progress"
                confidence = 0.7
            elif any(phrase in content_lower for phrase in ["error", "failed", "cannot", "unable"]):
                status = "failed"
                confidence = 0.3
            elif any(phrase in content_lower for phrase in ["needs revision", "needs work", "requires changes"]):
                status = "needs_revision"
                confidence = 0.5
            else:
                status = "in_progress"
                confidence = 0.6
            
            return AgentResponse(
                content=response_content,
                status=status,
                confidence=confidence,
                metadata={"execution_count": self.execution_count}
            )
            
        except Exception as e:
            return AgentResponse(
                content=f"Error in {self.name}: {str(e)}",
                status="failed",
                confidence=0.0,
                metadata={"error": str(e)}
            )


class CustomAgent(BaseAgent):
    """Agent with custom execution function"""
    
    def __init__(self, name: str, description: str, execute_func: Callable[[str, Dict], AgentResponse]):
        super().__init__(name, description)
        self.execute_func = execute_func
    
    def execute(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """Execute using custom function"""
        self.execution_count += 1
        return self.execute_func(task, context)


class TemplateAgent(BaseAgent):
    """Agent that uses string templates for simple responses"""
    
    def __init__(self, name: str, description: str, response_template: str, 
                 status_keywords: Dict[str, List[str]] = None):
        super().__init__(name, description)
        self.response_template = response_template
        self.status_keywords = status_keywords or {
            "complete": ["done", "finished", "completed", "ready"],
            "in_progress": ["working", "processing", "analyzing"],
            "failed": ["error", "failed", "cannot"]
        }
    
    def execute(self, task: str, context: Dict[str, Any]) -> AgentResponse:
        """Execute using template"""
        self.execution_count += 1
        
        try:
            # Format template with available variables
            response_content = self.response_template.format(
                task=task,
                context=context,
                agent_name=self.name,
                execution_count=self.execution_count,
                **context
            )
            
            # Determine status based on keywords
            content_lower = response_content.lower()
            status = "in_progress"
            confidence = 0.6
            
            for status_name, keywords in self.status_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    status = status_name
                    confidence = 0.8 if status_name == "complete" else 0.6
                    break
            
            return AgentResponse(
                content=response_content,
                status=status,
                confidence=confidence,
                metadata={"execution_count": self.execution_count}
            )
            
        except Exception as e:
            return AgentResponse(
                content=f"Template error in {self.name}: {str(e)}",
                status="failed",
                confidence=0.0,
                metadata={"error": str(e)}
            )


# Router Strategies
class BaseRouter(ABC):
    """Abstract base class for routing strategies"""
    
    @abstractmethod
    def decide_next_agent(self, state: OrchestratorState, agents: Dict[str, BaseAgent]) -> str:
        """Decide which agent should execute next"""
        pass


class LLMRouter(BaseRouter):
    """Router that uses any LLM for decision making"""
    
    def __init__(self, llm: Any):
        self.llm = llm
        self.system_prompt = """You are an intelligent Router Agent that coordinates work between specialized agents.

Available agents: {agent_descriptions}

Current situation:
- Task: {current_task}
- Iteration: {iteration}/{max_iterations}
- Agent Status: {agent_status}
- Recent Activity: {recent_messages}

Your job is to:
1. Analyze the current progress and agent statuses
2. Decide which agent should work next (or if work is complete)
3. Consider agent capabilities and current task requirements

Respond with JSON format:
{{
    "next_agent": "agent_name|end",
    "reasoning": "detailed explanation of your decision",
    "confidence": 0.85,
    "work_complete": true/false
}}"""
    
    def _invoke_llm(self, prompt: str) -> str:
        """Invoke LLM with flexible interface support"""
        if self.llm is None:
            raise ValueError("No LLM provided for router")
        
        try:
            # Try different LLM interfaces (same logic as LLMAgent)
            if hasattr(self.llm, 'invoke'):
                if hasattr(self.llm, 'bind') or 'langchain' in str(type(self.llm)):
                    from langchain_core.messages import HumanMessage, SystemMessage
                    messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
                    response = self.llm.invoke(messages)
                    return response.content if hasattr(response, 'content') else str(response)
                else:
                    response = self.llm.invoke(prompt)
                    return response.content if hasattr(response, 'content') else str(response)
            
            elif hasattr(self.llm, 'generate') or hasattr(self.llm, '__call__'):
                full_prompt = f"{self.system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
                response = self.llm(full_prompt)
                return str(response)
            
            elif hasattr(self.llm, 'chat') or hasattr(self.llm, 'complete'):
                if hasattr(self.llm, 'chat'):
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    response = self.llm.chat(messages=messages)
                else:
                    full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                    response = self.llm.complete(prompt=full_prompt)
                
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    return response.choices[0].message.content
                elif hasattr(response, 'content'):
                    return response.content
                else:
                    return str(response)
            
            else:
                full_prompt = f"{self.system_prompt}\n\nDecision needed for: {prompt}"
                return str(self.llm).format(prompt=full_prompt)
                
        except Exception as e:
            return f"Error invoking router LLM: {str(e)}"
    
    def decide_next_agent(self, state: OrchestratorState, agents: Dict[str, BaseAgent]) -> str:
        """Use LLM to decide next agent"""
        
        # Prepare agent descriptions
        agent_descriptions = {name: agent.description for name, agent in agents.items()}
        
        # Prepare context
        prompt = f"""Make your routing decision based on the current state:

Available Agents: {json.dumps(agent_descriptions, indent=2)}
Current Task: {state['current_task']}
Iteration: {state['iteration_count']}/{state['max_iterations']}
Agent Status: {json.dumps(state['agent_status'], indent=2)}
Recent Messages: {json.dumps(state['messages'][-3:], indent=2)}

Provide your decision in JSON format."""
        
        try:
            # Format the system prompt with context
            formatted_system_prompt = self.system_prompt.format(
                agent_descriptions=json.dumps(agent_descriptions, indent=2),
                current_task=state['current_task'],
                iteration=state['iteration_count'],
                max_iterations=state['max_iterations'],
                agent_status=json.dumps(state['agent_status'], indent=2),
                recent_messages=json.dumps(state['messages'][-3:], indent=2)
            )
            
            # Temporarily update system prompt
            old_prompt = self.system_prompt
            self.system_prompt = formatted_system_prompt
            
            # Get router decision
            response_content = self._invoke_llm("Make your routing decision.")
            
            # Restore system prompt
            self.system_prompt = old_prompt
            
            # Parse JSON response
            decision_data = json.loads(response_content)
            next_agent = decision_data.get("next_agent", "end")
            
            # Validate agent exists
            if next_agent != "end" and next_agent not in agents:
                next_agent = list(agents.keys())[0] if agents else "end"
            
            return next_agent
            
        except Exception as e:
            # Fallback decision
            if state["iteration_count"] >= state["max_iterations"]:
                return "end"
            
            # Find agent with lowest execution count
            if agents:
                min_agent = min(agents.values(), key=lambda a: a.execution_count)
                return min_agent.name
            else:
                return "end"


class RoundRobinRouter(BaseRouter):
    """Simple round-robin routing strategy"""
    
    def __init__(self):
        self.current_index = 0
    
    def decide_next_agent(self, state: OrchestratorState, agents: Dict[str, BaseAgent]) -> str:
        """Round-robin through agents"""
        
        if not agents:
            return "end"
        
        # Check if we should end
        if state["iteration_count"] >= state["max_iterations"]:
            return "end"
        
        # Check if all agents report complete
        all_complete = all(
            status.get("last_status") == "complete" 
            for status in state["agent_status"].values()
        )
        if all_complete and state["iteration_count"] > len(agents):
            return "end"
        
        # Round-robin selection
        agent_names = list(agents.keys())
        selected_agent = agent_names[self.current_index % len(agent_names)]
        self.current_index += 1
        
        return selected_agent


class PriorityRouter(BaseRouter):
    """Router that prioritizes agents based on their confidence and status"""
    
    def __init__(self, priority_order: List[str] = None):
        self.priority_order = priority_order or []
    
    def decide_next_agent(self, state: OrchestratorState, agents: Dict[str, BaseAgent]) -> str:
        """Route based on priority and agent status"""
        
        if not agents:
            return "end"
        
        # Check if we should end
        if state["iteration_count"] >= state["max_iterations"]:
            return "end"
        
        # Check if all agents report complete
        all_complete = all(
            status.get("last_status") == "complete" 
            for status in state["agent_status"].values()
        )
        if all_complete:
            return "end"
        
        # Find agents that haven't completed yet
        incomplete_agents = []
        for agent_name in agents.keys():
            agent_status = state["agent_status"].get(agent_name, {})
            if agent_status.get("last_status") != "complete":
                incomplete_agents.append(agent_name)
        
        if not incomplete_agents:
            return "end"
        
        # Prioritize based on priority order
        for agent_name in self.priority_order:
            if agent_name in incomplete_agents:
                return agent_name
        
        # If no priority match, return first incomplete agent
        return incomplete_agents[0]


# Main Orchestrator Class
class MultiAgentOrchestrator:
    """Main orchestrator class for managing multiple AI agents"""
    
    def __init__(self, 
                 agents: List[BaseAgent], 
                 router: BaseRouter = None,
                 llm: Any = None,
                 max_iterations: int = 20,
                 timeout_seconds: int = 300):
        """
        Initialize the orchestrator
        
        Args:
            agents: List of agent instances
            router: Router strategy (default: LLMRouter if llm provided, else RoundRobinRouter)
            llm: Language model for router and LLM agents
            max_iterations: Maximum number of iterations
            timeout_seconds: Maximum execution time
        """
        self.agents = {agent.name: agent for agent in agents}
        self.llm = llm
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        
        # Set up router
        if router is None:
            if llm is not None:
                self.router = LLMRouter(llm)
            else:
                self.router = RoundRobinRouter()
        else:
            self.router = router
        
        # Set up LLM for agents that need it
        for agent in self.agents.values():
            if isinstance(agent, LLMAgent) and agent.llm is None:
                agent.llm = self.llm
        
        self._setup_graph()
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the orchestrator"""
        self.agents[agent.name] = agent
        if isinstance(agent, LLMAgent) and agent.llm is None:
            agent.llm = self.llm
    
    def remove_agent(self, agent_name: str):
        """Remove an agent from the orchestrator"""
        if agent_name in self.agents:
            del self.agents[agent_name]
    
    def _setup_graph(self):
        """Setup the LangGraph workflow"""
        
        self.workflow = StateGraph(OrchestratorState)
        
        # Add router node
        self.workflow.add_node("router", self._router_node)
        
        # Add agent nodes
        for agent_name in self.agents.keys():
            self.workflow.add_node(agent_name, self._create_agent_node(agent_name))
        
        # Add edges
        self.workflow.add_edge(START, "router")
        
        # Conditional edges from router
        router_edges = {agent_name: agent_name for agent_name in self.agents.keys()}
        router_edges["end"] = END
        
        self.workflow.add_conditional_edges("router", self._route_decision, router_edges)
        
        # All agents return to router
        for agent_name in self.agents.keys():
            self.workflow.add_edge(agent_name, "router")
        
        # Compile the graph
        self.app = self.workflow.compile()
    
    def _router_node(self, state: OrchestratorState) -> OrchestratorState:
        """Router node implementation"""
        
        start_time = datetime.now()
        
        # Make routing decision
        next_agent = self.router.decide_next_agent(state, self.agents)
        
        # Update state
        state["router_decision"] = next_agent
        state["iteration_count"] += 1
        
        # Add trace
        trace_entry = {
            "step_id": str(uuid.uuid4()),
            "timestamp": start_time.isoformat(),
            "agent_name": "router",
            "action": "route_decision",
            "input_data": f"Iteration {state['iteration_count']}",
            "output_data": f"Next agent: {next_agent}",
            "status": "complete",
            "duration_ms": int((datetime.now() - start_time).total_seconds() * 1000)
        }
        
        state["messages"].append(trace_entry)
        
        return state
    
    def _create_agent_node(self, agent_name: str):
        """Create a node function for a specific agent"""
        
        def agent_node(state: OrchestratorState) -> OrchestratorState:
            start_time = datetime.now()
            agent = self.agents[agent_name]
            
            # Prepare context
            context = {
                "current_iteration": state["iteration_count"],
                "total_iterations": state["max_iterations"],
                "agent_status": state["agent_status"],
                "recent_messages": state["messages"][-5:],
                "metadata": state["metadata"]
            }
            
            # Execute agent
            response = agent.execute(state["current_task"], context)
            
            # Update agent status
            if agent_name not in state["agent_status"]:
                state["agent_status"][agent_name] = {}
            
            state["agent_status"][agent_name].update({
                "last_status": response.status,
                "last_confidence": response.confidence,
                "execution_count": agent.execution_count,
                "last_execution": datetime.now().isoformat()
            })
            
            # Add trace
            trace_entry = {
                "step_id": str(uuid.uuid4()),
                "timestamp": start_time.isoformat(),
                "agent_name": agent_name,
                "action": "execute",
                "input_data": state["current_task"][:200] + "..." if len(state["current_task"]) > 200 else state["current_task"],
                "output_data": response.content[:200] + "..." if len(response.content) > 200 else response.content,
                "status": response.status,
                "duration_ms": int((datetime.now() - start_time).total_seconds() * 1000),
                "metadata": response.metadata
            }
            
            state["messages"].append(trace_entry)
            
            # Update final response if agent completed successfully
            if response.status == "complete":
                state["final_response"] = response.content
            
            return state
        
        return agent_node
    
    def _route_decision(self, state: OrchestratorState) -> str:
        """Determine routing decision"""
        
        decision = state.get("router_decision", "end")
        
        # Validate decision
        if decision == "end" or state["iteration_count"] >= state["max_iterations"]:
            return "end"
        
        if decision in self.agents:
            return decision
        else:
            return "end"
    
    def invoke(self, user_task: str, metadata: Dict[str, Any] = None) -> OrchestrationResult:
        """
        Main entry point for task execution
        
        Args:
            user_task: The task to be executed
            metadata: Optional metadata for the task
        
        Returns:
            OrchestrationResult containing response and trace
        """
        
        start_time = datetime.now()
        
        # Initialize state
        initial_state = OrchestratorState(
            messages=[],
            current_task=user_task,
            agent_status={},
            iteration_count=0,
            max_iterations=self.max_iterations,
            router_decision="",
            final_response="",
            metadata=metadata or {}
        )
        
        try:
            # Execute workflow
            final_state = self.app.invoke(initial_state)
            
            # Calculate execution time
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Convert messages to ExecutionTrace objects
            execution_trace = []
            for msg in final_state["messages"]:
                if isinstance(msg, dict):
                    execution_trace.append(ExecutionTrace(
                        step_id=msg.get("step_id", str(uuid.uuid4())),
                        timestamp=datetime.fromisoformat(msg.get("timestamp", datetime.now().isoformat())),
                        agent_name=msg.get("agent_name", "unknown"),
                        action=msg.get("action", "unknown"),
                        input_data=msg.get("input_data", ""),
                        output_data=msg.get("output_data", ""),
                        status=msg.get("status", "unknown"),
                        duration_ms=msg.get("duration_ms", 0),
                        metadata=msg.get("metadata", {})
                    ))
            
            # Determine final response
            final_response = final_state.get("final_response", "")
            if not final_response and execution_trace:
                # Use the last agent's output if no explicit final response
                for trace in reversed(execution_trace):
                    if trace.agent_name != "router" and trace.output_data:
                        final_response = trace.output_data
                        break
            
            return OrchestrationResult(
                task=user_task,
                final_response=final_response,
                total_iterations=final_state["iteration_count"],
                execution_time_ms=execution_time,
                agent_status=final_state["agent_status"],
                execution_trace=execution_trace,
                metadata=final_state["metadata"],
                success=True
            )
            
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return OrchestrationResult(
                task=user_task,
                final_response="",
                total_iterations=0,
                execution_time_ms=execution_time,
                agent_status={},
                execution_trace=[],
                metadata=metadata or {},
                success=False,
                error_message=str(e)
            )
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        return {name: agent.get_status() for name, agent in self.agents.items()}


# Utility Functions for Easy Agent Creation
def create_llm_agent(name: str, description: str, system_prompt: str, llm: Any = None) -> LLMAgent:
    """Utility function to create an LLM agent"""
    return LLMAgent(name, description, system_prompt, llm)


def create_custom_agent(name: str, description: str, execute_func: Callable[[str, Dict], AgentResponse]) -> CustomAgent:
    """Utility function to create a custom agent"""
    return CustomAgent(name, description, execute_func)


def create_template_agent(name: str, description: str, response_template: str, 
                         status_keywords: Dict[str, List[str]] = None) -> TemplateAgent:
    """Utility function to create a template agent"""
    return TemplateAgent(name, description, response_template, status_keywords)


# Example Usage and Testing
def create_example_agents(llm=None):
    """Create example agents for testing with any LLM"""
    
    # Code Review Agent
    code_review_agent = create_llm_agent(
        name="code_reviewer",
        description="Reviews code quality, security, and best practices",
        system_prompt="""You are a Code Review Agent specializing in:
        - Code quality assessment
        - Security vulnerability detection
        - Best practices enforcement
        - Performance optimization suggestions
        
        Analyze the given code/task and provide detailed feedback.
        If code needs improvement, specify what needs to be fixed.
        Respond with 'COMPLETE' when code meets high standards.""",
        llm=llm
    )
    
    # QA Agent
    qa_agent = create_llm_agent(
        name="qa_tester",
        description="Tests functionality, identifies edge cases and bugs",
        system_prompt="""You are a QA Agent responsible for:
        - Functional testing
        - Edge case identification
        - Input validation testing
        - Integration testing scenarios
        
        Test the given functionality thoroughly.
        Identify bugs, missing features, or edge cases.
        Respond with 'COMPLETE' when all tests pass.""",
        llm=llm
    )
    
    # Software Development Agent
    software_agent = create_llm_agent(
        name="developer",
        description="Implements features, fixes bugs, writes code",
        system_prompt="""You are a Software Development Agent that:
        - Implements new features
        - Fixes identified bugs
        - Writes clean, maintainable code
        - Follows best practices
        
        Based on feedback from other agents, implement solutions.
        Write production-ready code with proper error handling.
        Respond with 'COMPLETE' when implementation is finished.""",
        llm=llm
    )
    
    return [code_review_agent, qa_agent, software_agent]


def create_template_example_agents():
    """Create example template agents that don't require LLM"""
    
    # Simple validator agent
    validator_agent = create_template_agent(
        name="validator",
        description="Validates task completion",
        response_template="Task '{task}' has been validated. Status: COMPLETE after {execution_count} validations.",
        status_keywords={
            "complete": ["complete", "validated", "approved"],
            "failed": ["invalid", "failed", "rejected"]
        }
    )
    
    # Documentation agent
    doc_agent = create_template_agent(
        name="documenter",
        description="Creates documentation for completed tasks",
        response_template="Documentation generated for: {task}\n\nExecution #{execution_count}\nStatus: COMPLETE\nDocumentation ready for review.",
        status_keywords={
            "complete": ["complete", "generated", "ready"],
            "in_progress": ["working", "generating", "processing"]
        }
    )
    
    return [validator_agent, doc_agent]


def create_custom_example_agents():
    """Create example custom agents"""
    
    def simple_analyzer(task: str, context: Dict) -> AgentResponse:
        """Simple task analyzer"""
        word_count = len(task.split())
        complexity = "high" if word_count > 20 else "medium" if word_count > 10 else "low"
        
        response = f"Task Analysis Complete:\n- Word count: {word_count}\n- Complexity: {complexity}\n- Ready for processing"
        
        return AgentResponse(
            content=response,
            status="complete",
            confidence=0.9,
            metadata={"word_count": word_count, "complexity": complexity}
        )
    
    def task_prioritizer(task: str, context: Dict) -> AgentResponse:
        """Task prioritization agent"""
        priority_keywords = ["urgent", "important", "critical", "asap"]
        priority = "high" if any(kw in task.lower() for kw in priority_keywords) else "normal"
        
        response = f"Task Prioritization Complete:\n- Priority: {priority}\n- Task can proceed with {priority} priority"
        
        return AgentResponse(
            content=response,
            status="complete", 
            confidence=0.8,
            metadata={"priority": priority}
        )
    
    analyzer_agent = create_custom_agent(
        name="analyzer",
        description="Analyzes task complexity and requirements",
        execute_func=simple_analyzer
    )
    
    prioritizer_agent = create_custom_agent(
        name="prioritizer", 
        description="Determines task priority and urgency",
        execute_func=task_prioritizer
    )
    
    return [analyzer_agent, prioritizer_agent]


def main():
    """Example usage of the LLM-agnostic library"""
    
    print("Multi-Agent Orchestration Library - Example Usage")
    print("=" * 50)
    
    # Example 1: Using template agents (no LLM required)
    print("\n1. Template Agents Example (No LLM required):")
    template_agents = create_template_example_agents()
    orchestrator1 = MultiAgentOrchestrator(
        agents=template_agents,
        max_iterations=5
    )
    
    result1 = orchestrator1.invoke("Create a simple validation process for user inputs")
    print(f"Final Response: {result1.final_response}")
    print(f"Execution Trace: {len(result1.execution_trace)} steps")
    
    # Example 2: Using custom agents
    print("\n2. Custom Agents Example:")
    custom_agents = create_custom_example_agents()
    orchestrator2 = MultiAgentOrchestrator(
        agents=custom_agents,
        max_iterations=5
    )
    
    result2 = orchestrator2.invoke("Urgent: Process critical user authentication system")
    print(f"Final Response: {result2.final_response}")
    print(f"Agent Status: {result2.agent_status}")
    
    # Example 3: Mixed agents with priority router
    print("\n3. Mixed Agents with Priority Router:")
    mixed_agents = template_agents + custom_agents
    priority_router = PriorityRouter(priority_order=["analyzer", "prioritizer", "validator", "documenter"])
    
    orchestrator3 = MultiAgentOrchestrator(
        agents=mixed_agents,
        router=priority_router,
        max_iterations=8
    )
    
    result3 = orchestrator3.invoke("Design and implement a secure payment processing system")
    print(f"Total Iterations: {result3.total_iterations}")
    print(f"Execution Time: {result3.execution_time_ms}ms")
    
    # Display detailed trace
    print(f"\nDetailed Execution Trace:")
    for i, trace in enumerate(result3.execution_trace, 1):
        print(f"  Step {i}: {trace.agent_name} -> {trace.action} [{trace.status}] ({trace.duration_ms}ms)")
    
    # Example 4: With LLM (if available)
    print("\n4. LLM Integration Example:")
    print("To use with LLM, simply pass any LLM instance:")
    print("""
    # With LangChain LLM
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(api_key="your-key")
    
    # With Hugging Face
    from transformers import pipeline
    llm = pipeline("text-generation", model="gpt2")
    
    # With any custom LLM
    class MyLLM:
        def invoke(self, messages):
            # Your LLM logic here
            return response
    
    llm = MyLLM()
    
    # Create agents with LLM
    agents = create_example_agents(llm)
    orchestrator = MultiAgentOrchestrator(agents, llm=llm)
    result = orchestrator.invoke("Your task here")
    """)
    
    print("\nLibrary supports multiple LLM interfaces:")
    print("- LangChain style: llm.invoke(messages)")  
    print("- OpenAI style: llm.chat(messages) or llm.complete(prompt)")
    print("- Callable: llm(prompt)")
    print("- Custom interfaces via flexible detection")


# Advanced Examples
class DatabaseAgent(CustomAgent):
    """Example of a specialized database agent"""
    
    def __init__(self):
        def db_execute(task: str, context: Dict) -> AgentResponse:
            # Simulate database operations
            if "create" in task.lower():
                return AgentResponse("Database table created successfully", "complete", 0.9)
            elif "query" in task.lower():
                return AgentResponse("Query executed, 42 rows returned", "complete", 0.95)
            elif "update" in task.lower():
                return AgentResponse("Update operation in progress", "in_progress", 0.7)
            else:
                return AgentResponse("Database operation not recognized", "failed", 0.3)
        
        super().__init__(
            name="database",
            description="Handles database operations and queries",
            execute_func=db_execute
        )


class SecurityAgent(LLMAgent):
    """Example of a specialized security agent"""
    
    def __init__(self, llm=None):
        super().__init__(
            name="security",
            description="Performs security analysis and vulnerability assessment",
            system_prompt="""You are a Security Agent focused on:
            - Vulnerability assessment
            - Security best practices
            - Access control evaluation
            - Risk analysis
            
            Analyze the given task for security implications and provide recommendations.
            Mark as COMPLETE when security review is finished.""",
            llm=llm
        )


def create_advanced_workflow_example():
    """Example of creating a complex workflow with specialized agents"""
    
    # Create specialized agents
    db_agent = DatabaseAgent()
    
    # Custom API agent
    def api_handler(task: str, context: Dict) -> AgentResponse:
        if "api" in task.lower() or "endpoint" in task.lower():
            return AgentResponse(
                "API endpoints configured and tested. All endpoints responding correctly.",
                "complete", 
                0.9
            )
        return AgentResponse("No API work required", "complete", 0.8)
    
    api_agent = create_custom_agent(
        "api_handler",
        "Manages API development and testing", 
        api_handler
    )
    
    # Deployment agent
    deploy_template = """Deployment Status for: {task}

Deployment Phase: {execution_count}/3
- Configuration: ✓
- Testing: {'✓' if execution_count > 1 else '⏳'}  
- Production: {'✓ COMPLETE' if execution_count > 2 else '⏳'}

Status: {'COMPLETE' if execution_count > 2 else 'IN_PROGRESS'}"""
    
    deploy_agent = create_template_agent(
        "deployer",
        "Handles application deployment",
        deploy_template
    )
    
    return [db_agent, api_agent, deploy_agent]


if __name__ == "__main__":
    main()
    
    print("\n" + "="*50)
    print("Advanced Workflow Example:")
    print("="*50)
    
    # Advanced workflow
    advanced_agents = create_advanced_workflow_example()
    advanced_orchestrator = MultiAgentOrchestrator(
        agents=advanced_agents,
        router=PriorityRouter(priority_order=["database", "api_handler", "deployer"]),
        max_iterations=10
    )
    
    result = advanced_orchestrator.invoke(
        "Build and deploy a REST API with database backend", 
        metadata={"project": "advanced_example", "environment": "production"}
    )
    
    print(f"Advanced Workflow Result:")
    print(f"- Success: {result.success}")
    print(f"- Final Response: {result.final_response}")
    print(f"- Total Steps: {len(result.execution_trace)}")
    print(f"- Metadata: {result.metadata}")

    #################
# Your specialized agents
agents = [
    backend_agent,     # Handles backend development
    frontend_agent,    # Handles UI/UX 
    database_agent,    # Handles data layer
    security_agent     # Handles security review
]

# One orchestrator manages them all
orchestrator = MultiAgentOrchestrator(agents, llm=your_llm)

# Execute complex tasks
result = orchestrator.invoke("Build a secure e-commerce platform")

# Get comprehensive results
print(f"Success: {result.success}")
print(f"Final deliverable: {result.final_response}")