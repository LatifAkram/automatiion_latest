import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  ExpandMore,
  PlayArrow,
  Storage,
  Psychology,
  Api,
  Web,
  DataUsage,
  Cloud,
  Email,
  Chat,
  Analytics,
  Security,
  Add,
  Info,
} from '@mui/icons-material';

interface ComponentPaletteProps {
  onAddNode: (nodeType: string, position: { x: number; y: number }) => void;
}

interface ComponentCategory {
  name: string;
  icon: React.ReactNode;
  components: Component[];
}

interface Component {
  type: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  category: string;
  parameters: string[];
}

const componentCategories: ComponentCategory[] = [
  {
    name: 'Automation Tasks',
    icon: <PlayArrow />,
    components: [
      {
        type: 'web_automation',
        name: 'Web Automation',
        description: 'Automate web browser interactions',
        icon: <Web />,
        category: 'automation',
        parameters: ['url', 'actions', 'selectors', 'wait_time'],
      },
      {
        type: 'api_integration',
        name: 'API Integration',
        description: 'Make API calls and process responses',
        icon: <Api />,
        category: 'automation',
        parameters: ['endpoint', 'method', 'headers', 'body', 'authentication'],
      },
      {
        type: 'data_processing',
        name: 'Data Processing',
        description: 'Transform and analyze data',
        icon: <DataUsage />,
        category: 'automation',
        parameters: ['input_data', 'transformation_rules', 'output_format'],
      },
      {
        type: 'file_operations',
        name: 'File Operations',
        description: 'Read, write, and manipulate files',
        icon: <Storage />,
        category: 'automation',
        parameters: ['file_path', 'operation', 'content', 'format'],
      },
    ],
  },
  {
    name: 'AI & Decision Making',
    icon: <Psychology />,
    components: [
      {
        type: 'ai_analysis',
        name: 'AI Analysis',
        description: 'Use AI to analyze data and make decisions',
        icon: <Analytics />,
        category: 'ai',
        parameters: ['input_data', 'ai_model', 'analysis_type', 'confidence_threshold'],
      },
      {
        type: 'decision_logic',
        name: 'Decision Logic',
        description: 'Implement conditional logic and branching',
        icon: <Psychology />,
        category: 'ai',
        parameters: ['conditions', 'true_action', 'false_action', 'evaluation_rules'],
      },
      {
        type: 'natural_language',
        name: 'Natural Language',
        description: 'Process and generate natural language',
        icon: <Chat />,
        category: 'ai',
        parameters: ['input_text', 'language_model', 'task_type', 'output_format'],
      },
    ],
  },
  {
    name: 'Data Sources',
    icon: <Storage />,
    components: [
      {
        type: 'database_connection',
        name: 'Database Connection',
        description: 'Connect to databases and execute queries',
        icon: <Storage />,
        category: 'data',
        parameters: ['connection_string', 'query', 'parameters', 'output_format'],
      },
      {
        type: 'cloud_storage',
        name: 'Cloud Storage',
        description: 'Access cloud storage services',
        icon: <Cloud />,
        category: 'data',
        parameters: ['service', 'bucket', 'path', 'operation', 'credentials'],
      },
      {
        type: 'email_integration',
        name: 'Email Integration',
        description: 'Send and receive emails',
        icon: <Email />,
        category: 'data',
        parameters: ['smtp_server', 'credentials', 'recipients', 'subject', 'body'],
      },
    ],
  },
  {
    name: 'Security & Compliance',
    icon: <Security />,
    components: [
      {
        type: 'authentication',
        name: 'Authentication',
        description: 'Handle user authentication and authorization',
        icon: <Security />,
        category: 'security',
        parameters: ['auth_type', 'credentials', 'permissions', 'session_management'],
      },
      {
        type: 'data_encryption',
        name: 'Data Encryption',
        description: 'Encrypt and decrypt sensitive data',
        icon: <Security />,
        category: 'security',
        parameters: ['encryption_algorithm', 'key_management', 'data_format'],
      },
      {
        type: 'audit_logging',
        name: 'Audit Logging',
        description: 'Log activities for compliance and security',
        icon: <Security />,
        category: 'security',
        parameters: ['log_level', 'log_format', 'retention_policy', 'destinations'],
      },
    ],
  },
];

export const ComponentPalette: React.FC<ComponentPaletteProps> = ({ onAddNode }) => {
  const [expanded, setExpanded] = useState<string | false>('Automation Tasks');

  const handleAccordionChange = (panel: string) => (
    event: React.SyntheticEvent,
    isExpanded: boolean
  ) => {
    setExpanded(isExpanded ? panel : false);
  };

  const handleAddComponent = (component: Component) => {
    // Calculate position for new node (center of the canvas)
    const position = {
      x: Math.random() * 400 + 100, // Random position within reasonable bounds
      y: Math.random() * 300 + 100,
    };

    onAddNode(component.type, position);
  };

  const getComponentIcon = (component: Component) => {
    return (
      <Box
        sx={{
          width: 40,
          height: 40,
          borderRadius: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'primary.light',
          color: 'primary.contrastText',
        }}
      >
        {component.icon}
      </Box>
    );
  };

  return (
    <Box sx={{ height: '100%', overflow: 'auto' }}>
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Component Palette
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Drag components to create your workflow
        </Typography>
      </Paper>

      {componentCategories.map((category) => (
        <Accordion
          key={category.name}
          expanded={expanded === category.name}
          onChange={handleAccordionChange(category.name)}
          sx={{ mb: 1 }}
        >
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {category.icon}
              <Typography variant="subtitle1">{category.name}</Typography>
              <Chip
                label={category.components.length}
                size="small"
                color="primary"
                variant="outlined"
              />
            </Box>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <List dense>
              {category.components.map((component) => (
                <ListItem
                  key={component.type}
                  sx={{
                    cursor: 'pointer',
                    '&:hover': {
                      bgcolor: 'action.hover',
                    },
                    borderBottom: '1px solid',
                    borderColor: 'divider',
                  }}
                  onClick={() => handleAddComponent(component)}
                >
                  <ListItemIcon>{getComponentIcon(component)}</ListItemIcon>
                  <ListItemText
                    primary={component.name}
                    secondary={
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          {component.description}
                        </Typography>
                        <Box sx={{ mt: 1 }}>
                          {component.parameters.slice(0, 3).map((param) => (
                            <Chip
                              key={param}
                              label={param}
                              size="small"
                              variant="outlined"
                              sx={{ mr: 0.5, mb: 0.5 }}
                            />
                          ))}
                          {component.parameters.length > 3 && (
                            <Chip
                              label={`+${component.parameters.length - 3} more`}
                              size="small"
                              variant="outlined"
                              color="secondary"
                            />
                          )}
                        </Box>
                      </Box>
                    }
                  />
                  <Tooltip title="Add to workflow">
                    <IconButton size="small" color="primary">
                      <Add />
                    </IconButton>
                  </Tooltip>
                </ListItem>
              ))}
            </List>
          </AccordionDetails>
        </Accordion>
      ))}

      <Paper sx={{ p: 2, mt: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Quick Actions
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Tooltip title="Add Web Automation Task">
            <Chip
              icon={<Web />}
              label="Web Task"
              onClick={() => handleAddComponent(componentCategories[0].components[0])}
              clickable
              color="primary"
              variant="outlined"
            />
          </Tooltip>
          <Tooltip title="Add API Integration">
            <Chip
              icon={<Api />}
              label="API Call"
              onClick={() => handleAddComponent(componentCategories[0].components[1])}
              clickable
              color="primary"
              variant="outlined"
            />
          </Tooltip>
          <Tooltip title="Add AI Analysis">
            <Chip
              icon={<Analytics />}
              label="AI Analysis"
              onClick={() => handleAddComponent(componentCategories[1].components[0])}
              clickable
              color="primary"
              variant="outlined"
            />
          </Tooltip>
          <Tooltip title="Add Decision Logic">
            <Chip
              icon={<Psychology />}
              label="Decision"
              onClick={() => handleAddComponent(componentCategories[1].components[1])}
              clickable
              color="primary"
              variant="outlined"
            />
          </Tooltip>
        </Box>
      </Paper>

      <Paper sx={{ p: 2, mt: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Tips
        </Typography>
        <Typography variant="body2" color="text.secondary">
          • Click any component to add it to your workflow
        </Typography>
        <Typography variant="body2" color="text.secondary">
          • Connect components by dragging from output to input
        </Typography>
        <Typography variant="body2" color="text.secondary">
          • Configure parameters in the properties panel
        </Typography>
        <Typography variant="body2" color="text.secondary">
          • Use AI components for intelligent decision making
        </Typography>
      </Paper>
    </Box>
  );
};