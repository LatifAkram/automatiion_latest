import React, { useState, useCallback, useRef } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  MiniMap,
  NodeTypes,
  EdgeTypes,
} from 'react-flow-renderer';
import { Box, Paper, Typography, Button, IconButton, Tooltip } from '@mui/material';
import {
  PlayArrow,
  Stop,
  Save,
  Add,
  Settings,
  BugReport,
  Timeline,
} from '@mui/icons-material';
import { toast } from 'react-hot-toast';

import { TaskNode } from './nodes/TaskNode';
import { DecisionNode } from './nodes/DecisionNode';
import { DataNode } from './nodes/DataNode';
import { CustomEdge } from './edges/CustomEdge';
import { ComponentPalette } from './ComponentPalette';
import { PropertiesPanel } from './PropertiesPanel';
import { ExecutionPanel } from './ExecutionPanel';
import { useWorkflowExecution } from '../hooks/useWorkflowExecution';
import { useWorkflowValidation } from '../hooks/useWorkflowValidation';

const nodeTypes: NodeTypes = {
  task: TaskNode,
  decision: DecisionNode,
  data: DataNode,
};

const edgeTypes: EdgeTypes = {
  custom: CustomEdge,
};

interface WorkflowDesignerProps {
  workflowId?: string;
  onSave?: (workflow: any) => void;
  onExecute?: (workflow: any) => void;
}

export const WorkflowDesigner: React.FC<WorkflowDesignerProps> = ({
  workflowId,
  onSave,
  onExecute,
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionHistory, setExecutionHistory] = useState<any[]>([]);
  
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const reactFlowInstance = useRef<any>(null);

  const { executeWorkflow, executionStatus } = useWorkflowExecution();
  const { validateWorkflow, validationErrors } = useWorkflowValidation();

  // Handle node selection
  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  // Handle edge connections
  const onConnect = useCallback(
    (params: Connection) => {
      // Validate connection
      const isValidConnection = validateConnection(params);
      if (!isValidConnection) {
        toast.error('Invalid connection between these nodes');
        return;
      }

      setEdges((eds) => addEdge({ ...params, type: 'custom' }, eds));
      toast.success('Connection created successfully');
    },
    [setEdges]
  );

  // Validate connection between nodes
  const validateConnection = (connection: Connection): boolean => {
    const sourceNode = nodes.find(node => node.id === connection.source);
    const targetNode = nodes.find(node => node.id === connection.target);
    
    if (!sourceNode || !targetNode) return false;
    
    // Prevent self-connection
    if (connection.source === connection.target) return false;
    
    // Validate node type compatibility
    const sourceType = sourceNode.type;
    const targetType = targetNode.type;
    
    // Data nodes can only connect to task nodes
    if (sourceType === 'data' && targetType !== 'task') return false;
    if (targetType === 'data' && sourceType !== 'task') return false;
    
    // Decision nodes can connect to task or data nodes
    if (sourceType === 'decision' && !['task', 'data'].includes(targetType)) return false;
    if (targetType === 'decision' && !['task', 'data'].includes(sourceType)) return false;
    
    return true;
  };

  // Add new node from palette
  const onAddNode = useCallback((nodeType: string, position: { x: number; y: number }) => {
    const newNode: Node = {
      id: `node_${Date.now()}`,
      type: nodeType as any,
      position,
      data: {
        label: `New ${nodeType}`,
        type: nodeType,
        parameters: {},
        config: {},
      },
    };

    setNodes((nds) => nds.concat(newNode));
    toast.success(`${nodeType} node added`);
  }, [setNodes]);

  // Update node data
  const onNodeDataChange = useCallback((nodeId: string, data: any) => {
    setNodes((nds) =>
      nds.map((node) =>
        node.id === nodeId ? { ...node, data: { ...node.data, ...data } } : node
      )
    );
  }, [setNodes]);

  // Save workflow
  const handleSave = useCallback(async () => {
    try {
      const workflow = {
        id: workflowId,
        nodes,
        edges,
        metadata: {
          name: 'Workflow',
          description: 'Automation workflow',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      };

      // Validate workflow before saving
      const validation = validateWorkflow(workflow);
      if (!validation.isValid) {
        toast.error('Workflow validation failed');
        return;
      }

      if (onSave) {
        onSave(workflow);
      } else {
        // Save to backend
        const response = await fetch('/api/workflows', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(workflow),
        });

        if (response.ok) {
          toast.success('Workflow saved successfully');
        } else {
          throw new Error('Failed to save workflow');
        }
      }
    } catch (error) {
      toast.error('Failed to save workflow');
      console.error('Save error:', error);
    }
  }, [workflowId, nodes, edges, onSave, validateWorkflow]);

  // Execute workflow
  const handleExecute = useCallback(async () => {
    try {
      setIsExecuting(true);
      
      const workflow = {
        nodes,
        edges,
        metadata: {
          name: 'Workflow',
          description: 'Automation workflow',
        },
      };

      // Validate workflow before execution
      const validation = validateWorkflow(workflow);
      if (!validation.isValid) {
        toast.error('Workflow validation failed');
        return;
      }

      if (onExecute) {
        onExecute(workflow);
      } else {
        // Execute via backend
        const result = await executeWorkflow(workflow);
        setExecutionHistory(prev => [...prev, result]);
        toast.success('Workflow executed successfully');
      }
    } catch (error) {
      toast.error('Workflow execution failed');
      console.error('Execution error:', error);
    } finally {
      setIsExecuting(false);
    }
  }, [nodes, edges, onExecute, validateWorkflow, executeWorkflow]);

  // Stop execution
  const handleStop = useCallback(() => {
    setIsExecuting(false);
    toast.success('Workflow execution stopped');
  }, []);

  // Clear workflow
  const handleClear = useCallback(() => {
    setNodes([]);
    setEdges([]);
    setSelectedNode(null);
    setExecutionHistory([]);
    toast.success('Workflow cleared');
  }, [setNodes, setEdges]);

  return (
    <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* Component Palette */}
      <Box sx={{ width: 250, borderRight: 1, borderColor: 'divider' }}>
        <ComponentPalette onAddNode={onAddNode} />
      </Box>

      {/* Main Design Area */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Toolbar */}
        <Paper sx={{ p: 1, borderBottom: 1, borderColor: 'divider' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="h6" sx={{ flex: 1 }}>
              Workflow Designer
            </Typography>
            
            <Tooltip title="Save Workflow">
              <IconButton onClick={handleSave} color="primary">
                <Save />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Execute Workflow">
              <IconButton 
                onClick={handleExecute} 
                disabled={isExecuting}
                color="success"
              >
                <PlayArrow />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Stop Execution">
              <IconButton 
                onClick={handleStop} 
                disabled={!isExecuting}
                color="error"
              >
                <Stop />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Debug Workflow">
              <IconButton color="info">
                <BugReport />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Execution History">
              <IconButton color="secondary">
                <Timeline />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Settings">
              <IconButton>
                <Settings />
              </IconButton>
            </Tooltip>
          </Box>
        </Paper>

        {/* Flow Canvas */}
        <Box sx={{ flex: 1, position: 'relative' }} ref={reactFlowWrapper}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            fitView
            attributionPosition="bottom-left"
          >
            <Controls />
            <Background />
            <MiniMap />
          </ReactFlow>
        </Box>
      </Box>

      {/* Properties Panel */}
      <Box sx={{ width: 300, borderLeft: 1, borderColor: 'divider' }}>
        <PropertiesPanel
          selectedNode={selectedNode}
          onNodeDataChange={onNodeDataChange}
          validationErrors={validationErrors}
        />
      </Box>

      {/* Execution Panel */}
      {isExecuting && (
        <ExecutionPanel
          executionStatus={executionStatus}
          executionHistory={executionHistory}
          onClose={() => setIsExecuting(false)}
        />
      )}
    </Box>
  );
};