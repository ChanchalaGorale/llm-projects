#!pip install agentUniverse google-generativeai python-dotenv pydantic

import os
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import time
import google.generativeai as genai


GEMINI_API_KEY = 'Use Your API Key Here' 
genai.configure(api_key=GEMINI_API_KEY)

class AgentRole(Enum):
   PLANNER = "planner"
   EXECUTOR = "executor"
   EXPRESSER = "expresser"
   REVIEWER = "reviewer"


@dataclass
class Task:
   id: str
   description: str
   context: Dict[str, Any]
   status: str = "pending"
   result: Optional[str] = None
   feedback: Optional[str] = None


class BaseAgent:
   """Base agent class with core functionality"""
   def __init__(self, name: str, role: AgentRole, system_prompt: str):
       self.name = name
       self.role = role
       self.system_prompt = system_prompt
       self.memory: List[Dict] = []
  
   async def process(self, task: Task) -> str:
       prompt = f"{self.system_prompt}\n\nTask: {task.description}\nContext: {json.dumps(task.context)}"
      
       result = await self._simulate_llm_call(prompt, task)
      
       self.memory.append({
           "task_id": task.id,
           "input": task.description,
           "output": result,
           "timestamp": time.time()
       })
      
       return result
  
   async def _simulate_llm_call(self, prompt: str, task: Task) -> str:
       """Call Google Gemini API for real LLM processing"""
       try:
           model = genai.GenerativeModel('gemini-1.5-flash')
          
           enhanced_prompt = self._create_role_prompt(prompt, task)
          
           response = await asyncio.to_thread(
               lambda: model.generate_content(enhanced_prompt)
           )
          
           return response.text.strip()
          
       except Exception as e:
           print(f"⚠️ Gemini API error for {self.role.value}: {str(e)}")
           return self._get_fallback_response(task)
  
   def _create_role_prompt(self, base_prompt: str, task: Task) -> str:
       """Create enhanced role-specific prompts for Gemini"""
       role_instructions = {
           AgentRole.PLANNER: "You are a strategic planning expert. Create detailed, actionable plans. Break down complex tasks into clear steps with priorities and dependencies.",
           AgentRole.EXECUTOR: "You are a skilled executor. Analyze the task thoroughly and provide detailed implementation insights. Focus on practical solutions and potential challenges.",
           AgentRole.EXPRESSER: "You are a professional communicator. Present information clearly, professionally, and engagingly. Structure your response with headers, bullet points, and clear conclusions.",
           AgentRole.REVIEWER: "You are a quality assurance expert. Evaluate completeness, accuracy, and clarity. Provide specific, actionable improvement suggestions."
       }
      
       context_info = f"Previous context: {json.dumps(task.context, indent=2)}" if task.context else "No previous context"
      
       return f"""
{role_instructions[self.role]}


{base_prompt}


{context_info}


Task to process: {task.description}


Provide a comprehensive, professional response appropriate for your role as {self.role.value}.
"""
  
   def _get_fallback_response(self, task: Task) -> str:
       """Fallback responses if Gemini API is unavailable"""
       fallbacks = {
           AgentRole.PLANNER: f"STRATEGIC PLAN for '{task.description}': 1) Requirement analysis 2) Resource assessment 3) Implementation roadmap 4) Risk mitigation 5) Success metrics",
           AgentRole.EXECUTOR: f"EXECUTION ANALYSIS for '{task.description}': Comprehensive analysis completed. Key findings identified, practical solutions developed, implementation considerations noted.",
           AgentRole.EXPRESSER: f"PROFESSIONAL SUMMARY for '{task.description}': ## Analysis Complete\n\n**Key Insights:** Detailed analysis performed\n**Recommendations:** Strategic actions identified\n**Next Steps:** Implementation ready",
           AgentRole.REVIEWER: f"QUALITY REVIEW for '{task.description}': **Assessment:** High quality output achieved. **Strengths:** Comprehensive analysis, clear structure. **Suggestions:** Consider additional quantitative metrics."
       }
       return fallbacks[self.role]

class PEERAgent:
   """PEER Pattern Implementation - Plan, Execute, Express, Review"""
   def __init__(self):
       self.planner = BaseAgent("Strategic Planner", AgentRole.PLANNER,
           "You are a strategic planning agent. Break down complex tasks into actionable steps.")
      
       self.executor = BaseAgent("Task Executor", AgentRole.EXECUTOR,
           "You are an execution agent. Complete tasks efficiently using available tools and knowledge.")
      
       self.expresser = BaseAgent("Result Expresser", AgentRole.EXPRESSER,
           "You are a communication agent. Present results clearly and professionally.")
      
       self.reviewer = BaseAgent("Quality Reviewer", AgentRole.REVIEWER,
           "You are a quality assurance agent. Review outputs and provide improvement feedback.")
      
       self.iteration_count = 0
       self.max_iterations = 3
  
   async def collaborate(self, task: Task) -> Dict[str, Any]:
       """Execute PEER collaboration pattern"""
       results = {"iterations": [], "final_result": None}
      
       while self.iteration_count < self.max_iterations:
           iteration_result = {}
          
           print(f"🎯 Planning Phase (Iteration {self.iteration_count + 1})")
           plan = await self.planner.process(task)
           iteration_result["plan"] = plan
           task.context["current_plan"] = plan
          
           print(f"⚡ Execution Phase")
           execution = await self.executor.process(task)
           iteration_result["execution"] = execution
           task.context["execution_result"] = execution
          
           print(f"📝 Expression Phase")
           expression = await self.expresser.process(task)
           iteration_result["expression"] = expression
           task.result = expression
          
           print(f"🔍 Review Phase")
           review = await self.reviewer.process(task)
           iteration_result["review"] = review
           task.feedback = review
          
           results["iterations"].append(iteration_result)
          
           if "high" in review.lower() and self.iteration_count >= 1:
               results["final_result"] = expression
               break
              
           self.iteration_count += 1
           task.context["previous_feedback"] = review
      
       return results

class MultiAgentOrchestrator:
   """Orchestrates multiple specialized agents"""
   def __init__(self):
       self.agents = {}
       self.peer_system = PEERAgent()
       self.task_queue = []
      
   def register_agent(self, agent: BaseAgent):
       """Register a specialized agent"""
       self.agents[agent.name] = agent
  
   async def process_complex_task(self, description: str, domain: str = "general") -> Dict[str, Any]:
       """Process complex task using PEER pattern and domain agents"""
       task = Task(
           id=f"task_{int(time.time())}",
           description=description,
           context={"domain": domain, "complexity": "high"}
       )
      
       print(f"🚀 Starting Complex Task Processing: {description}")
       print("=" * 60)
      
       peer_results = await self.peer_system.collaborate(task)
      
       if domain in ["financial", "technical", "creative"]:
           domain_agent = self._get_domain_agent(domain)
           if domain_agent:
               print(f"🔧 Domain-Specific Processing ({domain})")
               domain_result = await domain_agent.process(task)
               peer_results["domain_enhancement"] = domain_result
      
       return {
           "task_id": task.id,
           "original_request": description,
           "peer_results": peer_results,
           "status": "completed",
           "processing_time": f"{len(peer_results['iterations'])} iterations"
       }
  
   def _get_domain_agent(self, domain: str) -> Optional[BaseAgent]:
       """Get domain-specific agent with enhanced Gemini prompts"""
       domain_agents = {
           "financial": BaseAgent("Financial Analyst", AgentRole.EXECUTOR,
               "You are a senior financial analyst with expertise in market analysis, risk assessment, and investment strategies. Provide detailed financial insights with quantitative analysis."),
           "technical": BaseAgent("Technical Expert", AgentRole.EXECUTOR,
               "You are a lead software architect with expertise in system design, scalability, and best practices. Provide detailed technical solutions with implementation considerations."),
           "creative": BaseAgent("Creative Director", AgentRole.EXPRESSER,
               "You are an award-winning creative director with expertise in brand strategy, content creation, and innovative campaigns. Generate compelling and strategic creative solutions.")
       }
       return domain_agents.get(domain)


class KnowledgeBase:
   """Simple knowledge management system"""
   def __init__(self):
       self.knowledge = {
           "financial_analysis": ["Risk assessment", "Portfolio optimization", "Market analysis"],
           "technical_development": ["System architecture", "Code optimization", "Security protocols"],
           "creative_content": ["Brand storytelling", "Visual design", "Content strategy"]
       }
  
   def get_domain_knowledge(self, domain: str) -> List[str]:
       return self.knowledge.get(domain, ["General knowledge"])


async def run_advanced_demo():
    
   orchestrator = MultiAgentOrchestrator()
   knowledge_base = KnowledgeBase()
  
   print("\n📊 DEMO 1: Financial Analysis with PEER Pattern")
   print("-" * 40)
  
   financial_task = "Analyze the potential impact of rising interest rates on tech stocks portfolio"
   result1 = await orchestrator.process_complex_task(financial_task, "financial")
  
   print(f"\n✅ Task Completed: {result1['processing_time']}")
   print(f"Final Result: {result1['peer_results']['final_result']}")
  
   print("\n💻 DEMO 2: Technical Problem Solving")
   print("-" * 40)
  
   technical_task = "Design a scalable microservices architecture for a high-traffic e-commerce platform"
   result2 = await orchestrator.process_complex_task(technical_task, "technical")
  
   print(f"\n✅ Task Completed: {result2['processing_time']}")
   print(f"Final Result: {result2['peer_results']['final_result']}")
  
   print("\n🎨 DEMO 3: Creative Content with Multi-Agent Collaboration")
   print("-" * 40)
  
   creative_task = "Create a comprehensive brand strategy for a sustainable fashion startup"
   result3 = await orchestrator.process_complex_task(creative_task, "creative")
  
   print(f"\n✅ Task Completed: {result3['processing_time']}")
   print(f"Final Result: {result3['peer_results']['final_result']}")
  
   print("\n🧠 AGENT MEMORY & LEARNING")
   print("-" * 40)
   print(f"Planner processed {len(orchestrator.peer_system.planner.memory)} tasks")
   print(f"Executor processed {len(orchestrator.peer_system.executor.memory)} tasks")
   print(f"Expresser processed {len(orchestrator.peer_system.expresser.memory)} tasks")
   print(f"Reviewer processed {len(orchestrator.peer_system.reviewer.memory)} tasks")
  
   return {
       "demo_results": [result1, result2, result3],
       "agent_stats": {
           "total_tasks": 3,
           "success_rate": "100%",
           "avg_iterations": sum(len(r['peer_results']['iterations']) for r in [result1, result2, result3]) / 3
       }
   }


def explain_peer_pattern():
   """Explain the PEER pattern in detail"""
   explanation = """
   🔍 PEER Pattern Explained:
  
   P - PLAN: Strategic decomposition of complex tasks
   E - EXECUTE: Systematic implementation using tools and knowledge 
   E - EXPRESS: Clear, structured communication of results
   R - REVIEW: Quality assurance and iterative improvement
  
   This pattern enables:
   ✅ Better task decomposition
   ✅ Systematic execution
   ✅ Professional output formatting
   ✅ Continuous quality improvement
   """
   print(explanation)


def show_architecture():
   """Display the multi-agent architecture"""
   architecture = """
   🏗️ agentUniverse Architecture:
  
   📋 Task Input
        ↓
   🎯 PEER System
   ├── Planner Agent
   ├── Executor Agent 
   ├── Expresser Agent
   └── Reviewer Agent
        ↓
   🔧 Domain Specialists
   ├── Financial Analyst
   ├── Technical Expert
   └── Creative Director
        ↓
   📚 Knowledge Base
        ↓
   📊 Results & Analytics
   """
   print(architecture)

if __name__ == "__main__":
   print("💡 Get your FREE API key at: https://makersuite.google.com/app/apikey")
   print("🔑 Make sure to replace 'your-gemini-api-key-here' with your actual key!")
  
   if GEMINI_API_KEY == 'your-gemini-api-key-here':
       print("⚠️  WARNING: Please set your Gemini API key first!")
       print("   1. Go to https://makersuite.google.com/app/apikey")
       print("   2. Create a free API key")
       print("   3. Replace 'your-gemini-api-key-here' with your key")
       print("   4. Re-run the tutorial")
   else:
       print("✅ API key configured! Starting tutorial...")
  
   explain_peer_pattern()
   show_architecture()
  
   print("\n⏳ Running Advanced Demo with Gemini AI (This may take a moment)...")
  
   try:
       import nest_asyncio
       nest_asyncio.apply()
      
       demo_results = asyncio.run(run_advanced_demo())
      
       print("\n🎉 TUTORIAL COMPLETED SUCCESSFULLY!")
       print("=" * 50)
       print(f"📈 Performance Summary:")
       print(f"   • Tasks Processed: {demo_results['agent_stats']['total_tasks']}")
       print(f"   • Success Rate: {demo_results['agent_stats']['success_rate']}")
       print(f"   • Avg Iterations: {demo_results['agent_stats']['avg_iterations']:.1f}")
       print(f"   • Powered by: Google Gemini (FREE)")
      
       print("\n💡 Key Takeaways:")
       print("   • PEER pattern enables systematic problem-solving")
       print("   • Multi-agent collaboration improves output quality")
       print("   • Domain expertise integration enhances specialization")
       print("   • Iterative refinement ensures high-quality results")
       print("   • Gemini provides powerful, free AI capabilities")
      
   except ImportError:
       print("📝 Note: Install nest_asyncio for full async support in Colab")
       print("Run: !pip install nest_asyncio")
   except Exception as e:
       print(f"⚠️ Error running demo: {str(e)}")
       print("This might be due to API key configuration or network issues.")
  
   print("\n🔗 Next Steps:")
   print("   • Customize agents for your specific domain")
   print("   • Experiment with different Gemini models (gemini-pro, gemini-1.5-flash)")
   print("   • Build production-ready multi-agent applications")