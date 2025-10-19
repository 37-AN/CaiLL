/**
 * RL Training & Learning Dashboard Component
 *
 * This component monitors the reinforcement learning training process,
 * showing how the AI agents are learning and improving over time.
 */

'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import {
  Brain,
  TrendingUp,
  TrendingDown,
  Activity,
  Target,
  Zap,
  Award,
  BarChart3,
  LineChart,
  PlayCircle,
  PauseCircle,
  RotateCcw,
  Database,
  Cpu,
  Eye,
  AlertTriangle,
  CheckCircle2
} from 'lucide-react'

interface Agent {
  id: string
  name: string
  type: string
  status: 'training' | 'active' | 'paused' | 'converged'
  episode: number
  totalEpisodes: number
  avgReward: number
  recentReward: number
  winRate: number
  sharpeRatio: number
  trainingProgress: number
  lastUpdated: string
}

interface TrainingMetrics {
  totalSteps: number
  episodesCompleted: number
  avgEpisodeReward: number
  avgEpisodeLength: number
  explorationRate: number
  learningRate: number
  loss: number
  policyLoss: number
  valueLoss: number
  entropy: number
}

interface ExperienceBuffer {
  capacity: number
  currentSize: number
  utilizationPercent: number
  oldestSample: string
  newestSample: string
}

export default function RLTrainingView() {
  const [agents, setAgents] = useState<Agent[]>([
    {
      id: 'trend-agent',
      name: 'Trend Following Agent',
      type: 'PPO',
      status: 'training',
      episode: 1247,
      totalEpisodes: 5000,
      avgReward: 0.342,
      recentReward: 0.456,
      winRate: 58.3,
      sharpeRatio: 1.24,
      trainingProgress: 24.9,
      lastUpdated: '2 minutes ago'
    },
    {
      id: 'mean-reversion-agent',
      name: 'Mean Reversion Agent',
      type: 'A2C',
      status: 'training',
      episode: 892,
      totalEpisodes: 5000,
      avgReward: 0.198,
      recentReward: 0.312,
      winRate: 54.1,
      sharpeRatio: 0.92,
      trainingProgress: 17.8,
      lastUpdated: '1 minute ago'
    },
    {
      id: 'volatility-agent',
      name: 'Volatility Trading Agent',
      type: 'DQN',
      status: 'active',
      episode: 5000,
      totalEpisodes: 5000,
      avgReward: 0.521,
      recentReward: 0.498,
      winRate: 62.7,
      sharpeRatio: 1.58,
      trainingProgress: 100,
      lastUpdated: '5 minutes ago'
    },
    {
      id: 'momentum-agent',
      name: 'Momentum Agent',
      type: 'PPO',
      status: 'paused',
      episode: 2341,
      totalEpisodes: 5000,
      avgReward: 0.267,
      recentReward: 0.289,
      winRate: 56.2,
      sharpeRatio: 1.03,
      trainingProgress: 46.8,
      lastUpdated: '15 minutes ago'
    }
  ])

  const [systemMetrics, setSystemMetrics] = useState<TrainingMetrics>({
    totalSteps: 1847329,
    episodesCompleted: 9480,
    avgEpisodeReward: 0.329,
    avgEpisodeLength: 195,
    explorationRate: 0.15,
    learningRate: 0.0003,
    loss: 0.042,
    policyLoss: 0.028,
    valueLoss: 0.014,
    entropy: 0.67
  })

  const [experienceBuffer, setExperienceBuffer] = useState<ExperienceBuffer>({
    capacity: 100000,
    currentSize: 87234,
    utilizationPercent: 87.2,
    oldestSample: '3 days ago',
    newestSample: 'just now'
  })

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'training': return 'bg-blue-500'
      case 'active': return 'bg-green-500'
      case 'paused': return 'bg-yellow-500'
      case 'converged': return 'bg-purple-500'
      default: return 'bg-gray-500'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'training': return <Activity className="h-4 w-4" />
      case 'active': return <CheckCircle2 className="h-4 w-4" />
      case 'paused': return <PauseCircle className="h-4 w-4" />
      case 'converged': return <Target className="h-4 w-4" />
      default: return <AlertTriangle className="h-4 w-4" />
    }
  }

  return (
    <div className="space-y-6">
      {/* Header with Overall Status */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">RL Training Monitor</h2>
          <p className="text-muted-foreground">
            Real-time monitoring of reinforcement learning agent training
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <PauseCircle className="mr-2 h-4 w-4" />
            Pause All
          </Button>
          <Button variant="outline" size="sm">
            <RotateCcw className="mr-2 h-4 w-4" />
            Reset
          </Button>
        </div>
      </div>

      {/* System-Wide Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Steps</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemMetrics.totalSteps.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">
              Across all agents
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Episode Reward</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemMetrics.avgEpisodeReward.toFixed(3)}</div>
            <p className="text-xs text-green-600">
              +12.3% from last checkpoint
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Exploration Rate</CardTitle>
            <Eye className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(systemMetrics.explorationRate * 100).toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">
              Epsilon-greedy strategy
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Training Loss</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemMetrics.loss.toFixed(4)}</div>
            <p className="text-xs text-green-600">
              Converging ↓
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Agent Training Cards */}
      <Card>
        <CardHeader>
          <CardTitle>Active Training Agents</CardTitle>
          <CardDescription>
            Monitor individual agent learning progress and performance
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {agents.map((agent) => (
              <Card key={agent.id} className="border">
                <CardContent className="pt-6">
                  <div className="space-y-4">
                    {/* Agent Header */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`h-10 w-10 rounded-full ${getStatusColor(agent.status)} flex items-center justify-center text-white`}>
                          {getStatusIcon(agent.status)}
                        </div>
                        <div>
                          <h4 className="font-semibold">{agent.name}</h4>
                          <p className="text-sm text-muted-foreground">
                            Algorithm: {agent.type} | Episode {agent.episode.toLocaleString()} / {agent.totalEpisodes.toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <Badge variant={agent.status === 'active' ? 'default' : 'secondary'}>
                        {agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}
                      </Badge>
                    </div>

                    {/* Training Progress Bar */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Training Progress</span>
                        <span className="font-medium">{agent.trainingProgress.toFixed(1)}%</span>
                      </div>
                      <Progress value={agent.trainingProgress} className="h-2" />
                    </div>

                    {/* Performance Metrics Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">Avg Reward</p>
                        <p className="text-lg font-bold">{agent.avgReward.toFixed(3)}</p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">Recent Reward</p>
                        <p className="text-lg font-bold flex items-center gap-1">
                          {agent.recentReward.toFixed(3)}
                          {agent.recentReward > agent.avgReward ? (
                            <TrendingUp className="h-4 w-4 text-green-600" />
                          ) : (
                            <TrendingDown className="h-4 w-4 text-red-600" />
                          )}
                        </p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">Win Rate</p>
                        <p className="text-lg font-bold">{agent.winRate.toFixed(1)}%</p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-xs text-muted-foreground">Sharpe Ratio</p>
                        <p className="text-lg font-bold">{agent.sharpeRatio.toFixed(2)}</p>
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-2 pt-2">
                      <Button variant="outline" size="sm">
                        <Eye className="mr-2 h-4 w-4" />
                        View Details
                      </Button>
                      {agent.status === 'training' ? (
                        <Button variant="outline" size="sm">
                          <PauseCircle className="mr-2 h-4 w-4" />
                          Pause
                        </Button>
                      ) : (
                        <Button variant="outline" size="sm">
                          <PlayCircle className="mr-2 h-4 w-4" />
                          Resume
                        </Button>
                      )}
                      <Button variant="outline" size="sm">
                        <Database className="mr-2 h-4 w-4" />
                        Export Model
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Experience Replay Buffer & Learning Metrics */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Experience Replay Buffer</CardTitle>
            <CardDescription>Memory storage for training samples</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground">Utilization</span>
                <span className="font-medium">
                  {experienceBuffer.currentSize.toLocaleString()} / {experienceBuffer.capacity.toLocaleString()}
                </span>
              </div>
              <Progress value={experienceBuffer.utilizationPercent} className="h-2" />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Oldest Sample</p>
                <p className="font-medium">{experienceBuffer.oldestSample}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Newest Sample</p>
                <p className="font-medium">{experienceBuffer.newestSample}</p>
              </div>
            </div>

            {experienceBuffer.utilizationPercent > 90 && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Buffer Nearly Full</AlertTitle>
                <AlertDescription>
                  Consider increasing capacity or enabling sample pruning
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Learning Hyperparameters</CardTitle>
            <CardDescription>Current training configuration</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Learning Rate</span>
                <span className="font-mono text-sm font-medium">{systemMetrics.learningRate}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Policy Loss</span>
                <span className="font-mono text-sm font-medium">{systemMetrics.policyLoss.toFixed(4)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Value Loss</span>
                <span className="font-mono text-sm font-medium">{systemMetrics.valueLoss.toFixed(4)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Entropy Coefficient</span>
                <span className="font-mono text-sm font-medium">{systemMetrics.entropy.toFixed(2)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Avg Episode Length</span>
                <span className="font-mono text-sm font-medium">{systemMetrics.avgEpisodeLength} steps</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Learning Progress Alert */}
      <Alert>
        <Brain className="h-4 w-4" />
        <AlertTitle>Continuous Learning Active</AlertTitle>
        <AlertDescription>
          Agents are learning from live market data and improving their strategies in real-time.
          Training checkpoints are saved every 500 episodes. Current session started 3 hours ago.
        </AlertDescription>
      </Alert>
    </div>
  )
}
