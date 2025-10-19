'use client'

import dynamic from 'next/dynamic'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Activity, Brain, BarChart3, Settings } from 'lucide-react'

// Dynamically import dashboard components to avoid SSR issues
const TradingView = dynamic(() => import('@/frontend/dashboard/trading_view'), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-96">Loading Trading View...</div>
})

const RLTrainingView = dynamic(() => import('@/frontend/dashboard/rl_training_view'), {
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-96">Loading RL Training View...</div>
})

export default function Home() {
  return (
    <div className="container mx-auto p-4 md:p-8 max-w-7xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">CaiLL AI Trading System</h1>
        <p className="text-muted-foreground">
          Reinforcement Learning-Powered Trading Platform with Educational Modules
        </p>
      </div>

      {/* Main Dashboard Tabs */}
      <Tabs defaultValue="trading" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4 lg:w-[600px]">
          <TabsTrigger value="trading" className="flex items-center gap-2">
            <Activity className="h-4 w-4" />
            <span className="hidden sm:inline">Trading</span>
          </TabsTrigger>
          <TabsTrigger value="learning" className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            <span className="hidden sm:inline">RL Training</span>
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            <span className="hidden sm:inline">Analytics</span>
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            <span className="hidden sm:inline">Settings</span>
          </TabsTrigger>
        </TabsList>

        {/* Trading View */}
        <TabsContent value="trading" className="space-y-4">
          <TradingView />
        </TabsContent>

        {/* RL Training View */}
        <TabsContent value="learning" className="space-y-4">
          <RLTrainingView />
        </TabsContent>

        {/* Analytics View */}
        <TabsContent value="analytics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Analytics & Performance</CardTitle>
              <CardDescription>
                Backtesting results, performance metrics, and strategy validation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-12 text-muted-foreground">
                <BarChart3 className="h-16 w-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg mb-2">Analytics Dashboard</p>
                <p className="text-sm">
                  Connect the Python backend to view backtesting results and performance metrics
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Settings View */}
        <TabsContent value="settings" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>System Settings</CardTitle>
              <CardDescription>
                Configure trading parameters, risk limits, and API connections
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Trading Mode</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground mb-2">Current: Paper Trading</p>
                      <p className="text-xs text-amber-600">
                        Switch to live trading only after successful paper trading period
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Backend Status</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground mb-2">Status: Disconnected</p>
                      <p className="text-xs text-muted-foreground">
                        Start the Python backend to enable trading features
                      </p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Risk Management</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground mb-2">Max Drawdown: 20%</p>
                      <p className="text-sm text-muted-foreground mb-2">Position Size Limit: 5%</p>
                      <p className="text-xs text-green-600">Risk controls active</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">API Keys</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground mb-2">Alpaca: Not configured</p>
                      <p className="text-sm text-muted-foreground mb-2">Pinecone: Not configured</p>
                      <p className="text-xs text-muted-foreground">
                        Configure in .env file
                      </p>
                    </CardContent>
                  </Card>
                </div>

                <Card className="border-amber-200 bg-amber-50 dark:bg-amber-950 dark:border-amber-800">
                  <CardHeader>
                    <CardTitle className="text-base flex items-center gap-2">
                      <Activity className="h-4 w-4" />
                      Getting Started
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="text-sm space-y-2">
                    <p>To fully activate the trading system:</p>
                    <ol className="list-decimal list-inside space-y-1 ml-2">
                      <li>Start the Python backend: <code className="bg-background px-1 py-0.5 rounded">python backend/main.py</code></li>
                      <li>Configure API keys in <code className="bg-background px-1 py-0.5 rounded">.env</code> file</li>
                      <li>Review risk management settings</li>
                      <li>Start with paper trading mode</li>
                    </ol>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Footer Info */}
      <div className="mt-8 p-4 border rounded-lg bg-muted/50">
        <p className="text-sm text-muted-foreground text-center">
          <strong>Phase 5 Complete:</strong> Backtesting & Strategy Validation |
          <strong className="ml-2">Next Phase:</strong> Educational System & Full Integration
        </p>
      </div>
    </div>
  )
}
