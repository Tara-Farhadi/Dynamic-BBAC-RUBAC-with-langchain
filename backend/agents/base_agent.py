"""
Base Agent class for GUARDIAN multi-agent system.
Provides common interface and functionality for all agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import asyncio


class BaseAgent(ABC):
    """
    Base class for all GUARDIAN agents.
    
    Each agent operates autonomously and can be invoked on-demand.
    Agents communicate through well-defined interfaces without tight coupling.
    """
    
    def __init__(self, agent_id: str, name: str):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable agent name
        """
        self.agent_id = agent_id
        self.name = name
        self.is_ready = False
        self.last_invocation: Optional[datetime] = None
        self.total_invocations = 0
        self.logger = logging.getLogger(f"guardian.{agent_id}")
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method to be implemented by each agent.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Processed output data
        """
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the agent and its sub-agents.
        
        Returns:
            True if initialization successful
        """
        pass
    
    async def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the agent with input data.
        Handles logging, timing, and error handling.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Agent output with metadata
        """
        start_time = datetime.utcnow()
        self.total_invocations += 1
        
        try:
            self.logger.info(f"Agent {self.name} invoked")
            
            # Ensure agent is initialized
            if not self.is_ready:
                self.initialize()
            
            # Process the input
            result = await self.process(input_data)
            
            # Add metadata
            end_time = datetime.utcnow()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            result['_agent_metadata'] = {
                'agent_id': self.agent_id,
                'agent_name': self.name,
                'processing_time_ms': round(processing_time_ms, 2),
                'timestamp': end_time.isoformat(),
                'invocation_count': self.total_invocations
            }
            
            self.last_invocation = end_time
            self.logger.info(f"Agent {self.name} completed in {processing_time_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Agent {self.name} error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                '_agent_metadata': {
                    'agent_id': self.agent_id,
                    'agent_name': self.name,
                    'error': True
                }
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the agent.
        
        Returns:
            Status information
        """
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'is_ready': self.is_ready,
            'last_invocation': self.last_invocation.isoformat() if self.last_invocation else None,
            'total_invocations': self.total_invocations
        }


class BaseSubAgent(ABC):
    """
    Base class for sub-agents within main agents.
    Sub-agents handle specific specialized tasks.
    """
    
    def __init__(self, name: str, parent_agent_id: str):
        """
        Initialize sub-agent.
        
        Args:
            name: Sub-agent name
            parent_agent_id: ID of parent agent
        """
        self.name = name
        self.parent_agent_id = parent_agent_id
        self.logger = logging.getLogger(f"guardian.{parent_agent_id}.{name}")
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the sub-agent's task.
        
        Args:
            input_data: Input data
            
        Returns:
            Execution result
        """
        pass
