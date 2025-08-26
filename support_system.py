import os
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Context model to store user information
class UserContext(BaseModel):
    name: str = "Guest"
    is_premium_user: bool = False
    issue_type: Optional[str] = None
    previous_queries: List[str] = Field(default_factory=list)

# Define issue types
class IssueType(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"

# Base Agent Class
class SupportAgent:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.context = None
    
    def set_context(self, context: UserContext):
        self.context = context
    
    def process_query(self, query: str) -> str:
        raise NotImplementedError("Subclasses must implement process_query")
    
    def get_available_tools(self) -> List[str]:
        raise NotImplementedError("Subclasses must implement get_available_tools")

# Triage Agent
class TriageAgent(SupportAgent):
    def __init__(self):
        super().__init__(
            "Triage Agent",
            "Routes user queries to the appropriate specialized agent"
        )
    
    def process_query(self, query: str) -> Dict[str, Any]:
        # Use Gemini to determine the issue type
        prompt = f"""
        Analyze the following user query and determine the most appropriate category:
        "{query}"
        
        Categories:
        - billing: Issues related to payments, refunds, subscriptions, invoices
        - technical: Problems with service, bugs, errors, functionality issues
        - general: Other inquiries, account questions, general information
        
        Respond with ONLY the category name (billing, technical, or general).
        """
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            issue_type = response.text.strip().lower()
            
            # Validate response
            if issue_type not in [it.value for it in IssueType]:
                issue_type = IssueType.GENERAL.value
        except Exception:
            issue_type = IssueType.GENERAL.value
        
        # Update context
        if self.context:
            self.context.issue_type = issue_type
            self.context.previous_queries.append(query)
        
        return {
            "issue_type": issue_type,
            "message": f"Your query has been categorized as: {issue_type}. Transferring you to the {issue_type} specialist."
        }

# Billing Agent
class BillingAgent(SupportAgent):
    def __init__(self):
        super().__init__(
            "Billing Agent",
            "Handles billing-related inquiries including refunds and payments"
        )
        self.tools = {
            "process_refund": {
                "function": self.process_refund,
                "description": "Process a refund for the user"
            },
            "explain_charges": {
                "function": self.explain_charges,
                "description": "Explain recent charges on the user's account"
            },
            "update_subscription": {
                "function": self.update_subscription,
                "description": "Update the user's subscription plan"
            }
        }
    
    def get_available_tools(self) -> List[str]:
        available_tools = []
        for tool_name, tool_info in self.tools.items():
            # Check if tool is enabled based on context
            if self.is_tool_enabled(tool_name):
                available_tools.append(tool_name)
        return available_tools
    
    def is_tool_enabled(self, tool_name: str) -> bool:
        if tool_name == "process_refund":
            return self.context and self.context.is_premium_user
        return True
    
    def process_query(self, query: str) -> str:
        # Use Gemini to determine which tool to use
        available_tools = self.get_available_tools()
        
        prompt = f"""
        You are a billing support agent. Based on the user query: "{query}"
        
        Available tools: {', '.join(available_tools)}
        
        Determine the most appropriate tool to use or provide a general response.
        If using a tool, respond with: TOOL:<tool_name>
        Otherwise, provide a helpful response to the user's billing inquiry.
        """
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text.startswith("TOOL:"):
                tool_name = response_text.split(":")[1].strip()
                if tool_name in self.tools and self.is_tool_enabled(tool_name):
                    tool_response = self.tools[tool_name]["function"](query)
                    return f"Used {tool_name}: {tool_response}"
                else:
                    return "I'm sorry, that action is not available for your account."
            else:
                return response_text
        except Exception as e:
            return f"I apologize, I encountered an error processing your billing request: {str(e)}"
    
    def process_refund(self, query: str) -> str:
        # Simulate refund processing
        return "Your refund has been processed successfully. It will reflect in your account within 5-7 business days."
    
    def explain_charges(self, query: str) -> str:
        # Simulate charge explanation
        return "Your recent charge of $49.99 is for your monthly subscription. This includes access to all premium features."
    
    def update_subscription(self, query: str) -> str:
        # Simulate subscription update
        return "Your subscription has been updated successfully. Your new plan will take effect in the next billing cycle."

# Technical Agent
class TechnicalAgent(SupportAgent):
    def __init__(self):
        super().__init__(
            "Technical Agent",
            "Handles technical support issues and service problems"
        )
        self.tools = {
            "restart_service": {
                "function": self.restart_service,
                "description": "Restart a service for the user"
            },
            "reset_password": {
                "function": self.reset_password,
                "description": "Reset the user's password"
            },
            "check_status": {
                "function": self.check_status,
                "description": "Check the status of services"
            }
        }
    
    def get_available_tools(self) -> List[str]:
        available_tools = []
        for tool_name, tool_info in self.tools.items():
            if self.is_tool_enabled(tool_name):
                available_tools.append(tool_name)
        return available_tools
    
    def is_tool_enabled(self, tool_name: str) -> bool:
        if tool_name == "restart_service":
            return self.context and self.context.issue_type == IssueType.TECHNICAL.value
        return True
    
    def process_query(self, query: str) -> str:
        available_tools = self.get_available_tools()
        
        prompt = f"""
        You are a technical support agent. Based on the user query: "{query}"
        
        Available tools: {', '.join(available_tools)}
        
        Determine the most appropriate tool to use or provide a general response.
        If using a tool, respond with: TOOL:<tool_name>
        Otherwise, provide a helpful response to the user's technical inquiry.
        """
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text.startswith("TOOL:"):
                tool_name = response_text.split(":")[1].strip()
                if tool_name in self.tools and self.is_tool_enabled(tool_name):
                    tool_response = self.tools[tool_name]["function"](query)
                    return f"Used {tool_name}: {tool_response}"
                else:
                    return "I'm sorry, that action is not available for your account or issue type."
            else:
                return response_text
        except Exception as e:
            return f"I apologize, I encountered an error processing your technical request: {str(e)}"
    
    def restart_service(self, query: str) -> str:
        # Simulate service restart
        return "The service has been restarted successfully. Please allow a few minutes for it to become fully operational."
    
    def reset_password(self, query: str) -> str:
        # Simulate password reset
        return "A password reset link has been sent to your email. Please check your inbox and follow the instructions."
    
    def check_status(self, query: str) -> str:
        # Simulate status check
        return "All systems are operational. No outages reported at this time."

# General Agent
class GeneralAgent(SupportAgent):
    def __init__(self):
        super().__init__(
            "General Agent",
            "Handles general inquiries and account questions"
        )
        self.tools = {
            "provide_info": {
                "function": self.provide_info,
                "description": "Provide general information about services"
            },
            "escalate_issue": {
                "function": self.escalate_issue,
                "description": "Escalate the issue to a human representative"
            }
        }
    
    def get_available_tools(self) -> List[str]:
        return list(self.tools.keys())
    
    def process_query(self, query: str) -> str:
        available_tools = self.get_available_tools()
        
        prompt = f"""
        You are a general support agent. Based on the user query: "{query}"
        
        Available tools: {', '.join(available_tools)}
        
        Determine the most appropriate tool to use or provide a general response.
        If using a tool, respond with: TOOL:<tool_name>
        Otherwise, provide a helpful response to the user's general inquiry.
        """
        
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text.startswith("TOOL:"):
                tool_name = response_text.split(":")[1].strip()
                if tool_name in self.tools:
                    tool_response = self.tools[tool_name]["function"](query)
                    return f"Used {tool_name}: {tool_response}"
                else:
                    return "I'm sorry, that action is not available."
            else:
                return response_text
        except Exception as e:
            return f"I apologize, I encountered an error processing your request: {str(e)}"
    
    def provide_info(self, query: str) -> str:
        # Simulate providing information
        return "Our company provides a range of services designed to meet your needs. For more specific information, please visit our website or contact our support team."
    
    def escalate_issue(self, query: str) -> str:
        # Simulate escalation
        return "Your issue has been escalated to a human representative. They will contact you within 24 hours."

# Support System
class SupportSystem:
    def __init__(self):
        self.agents = {
            "triage": TriageAgent(),
            "billing": BillingAgent(),
            "technical": TechnicalAgent(),
            "general": GeneralAgent()
        }
        self.current_agent = "triage"
        self.context = UserContext()
        
        # Set context for all agents
        for agent in self.agents.values():
            agent.set_context(self.context)
    
    def start(self):
        print("🔵 Welcome to the Support Agent System!")
        print("🔵 Type 'quit' to exit at any time.")
        print("🔵 Please describe your issue and we'll assist you.\n")
        
        while True:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("🔵 Thank you for using our support system. Goodbye!")
                break
            
            # Process the query with the current agent
            agent = self.agents[self.current_agent]
            response = agent.process_query(user_input)
            
            # Handle triage agent response (which includes handoff)
            if self.current_agent == "triage":
                result = response
                issue_type = result["issue_type"]
                print(f"🟢 {self.agents['triage'].name}: {result['message']}")
                
                # Handoff to the appropriate agent
                if issue_type == IssueType.BILLING.value:
                    self.current_agent = "billing"
                elif issue_type == IssueType.TECHNICAL.value:
                    self.current_agent = "technical"
                else:
                    self.current_agent = "general"
            else:
                # Specialized agent response
                print(f"🟢 {agent.name}: {response}")
                
                # Ask if user needs further assistance
                print("🔵 Do you have any other questions? (yes/no)")
                continue_session = input("👤 You: ").strip().lower()
                
                if continue_session in ['no', 'n', 'quit', 'exit']:
                    print("🔵 Thank you for using our support system. Goodbye!")
                    break
                else:
                    print("🔵 How can I help you further?")
        
        # Print session summary
        print(f"\n📊 Session Summary:")
        print(f"   User: {self.context.name}")
        print(f"   Premium: {'Yes' if self.context.is_premium_user else 'No'}")
        print(f"   Issue Type: {self.context.issue_type}")
        print(f"   Queries Handled: {len(self.context.previous_queries)}")

# Main execution
if __name__ == "__main__":
    # Initialize and start the support system
    support_system = SupportSystem()
    
    # Set up user context (in a real system, this would come from a database)
    support_system.context.name = "Aamna"
    support_system.context.is_premium_user = True  # Change to False to test refund restriction
    
    # Update context for all agents
    for agent in support_system.agents.values():
        agent.set_context(support_system.context)
    
    # Start the support session
    support_system.start()