/**
 * Trading View Dashboard Component
 * 
 * This component provides a comprehensive trading interface with real-time data,
 * portfolio overview, active positions, and trading controls.
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
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3,
  PieChart,
  Settings,
  Play,
  Pause,
  Square
} from 'lucide-react'

interface Position {
  id: string
  symbol: string
  type: 'LONG' | 'SHORT'
  quantity: number
  entry_price: number
  current_price: number
  unrealized_pnl: number
  unrealized_pnl_percent: number
  day_change: number
  day_change_percent: number
}

interface Portfolio {
  total_value: number
  cash_balance: number
  invested_value: number
  total_pnl: number
  total_pnl_percent: number
  day_change: number
  day_change_percent: number
}

interface MarketData {
  symbol: string
  price: number
  change: number
  change_percent: number
  volume: number
  market_cap: number
  last_updated: string
}

interface TradingSignal {
  id: string
  symbol: string
  action: 'BUY' | 'SELL' | 'HOLD'
  confidence: number
  reason: string
  timestamp: string
  executed: boolean
}

export default function TradingView() {
  const [portfolio, setPortfolio] = useState<Portfolio>({
    total_value: 125000,
    cash_balance: 25000,
    invested_value: 100000,
    total_pnl: 5000,
    total_pnl_percent: 4.2,
    day_change: 1200,
    day_change_percent: 0.97
  })

  const [positions, setPositions] = useState<Position[]>([
    {
      id: '1',
      symbol: 'AAPL',
      type: 'LONG',
      quantity: 100,
      entry_price: 145.50,
      current_price: 148.25,
      unrealized_pnl: 275,
      unrealized_pnl_percent: 1.89,
      day_change: 2.75,
      day_change_percent: 1.89
    },
    {
      id: '2',
      symbol: 'MSFT',
      type: 'LONG',
      quantity: 50,
      entry_price: 380.20,
      current_price: 378.90,
      unrealized_pnl: -65,
      unrealized_pnl_percent: -0.34,
      day_change: -1.30,
      day_change_percent: -0.34
    },
    {
      id: '3',
      symbol: 'TSLA',
      type: 'SHORT',
      quantity: 25,
      entry_price: 245.80,
      current_price: 242.15,
      unrealized_pnl: 91.25,
      unrealized_pnl_percent: 1.48,
      day_change: 3.65,
      day_change_percent: 1.48
    }
  ])

  const [marketData, setMarketData] = useState<MarketData[]>([
    {
      symbol: 'AAPL',
      price: 148.25,
      change: 2.75,
      change_percent: 1.89,
      volume: 52341567,
      market_cap: 2890000000000,
      last_updated: new Date().toISOString()
    },
    {
      symbol: 'MSFT',
      price: 378.90,
      change: -1.30,
      change_percent: -0.34,
      volume: 19876543,
      market_cap: 2810000000000,
      last_updated: new Date().toISOString()
    },
    {
      symbol: 'TSLA',
      price: 242.15,
      change: -3.65,
      change_percent: -1.48,
      volume: 87654321,
      market_cap: 770000000000,
      last_updated: new Date().toISOString()
    }
  ])

  const [tradingSignals, setTradingSignals] = useState<TradingSignal[]>([
    {
      id: '1',
      symbol: 'NVDA',
      action: 'BUY',
      confidence: 0.85,
      reason: 'Strong momentum detected with RSI oversold conditions',
      timestamp: new Date().toISOString(),
      executed: false
    },
    {
      id: '2',
      symbol: 'META',
      action: 'SELL',
      confidence: 0.72,
      reason: 'Resistance level reached with bearish divergence',
      timestamp: new Date().toISOString(),
      executed: false
    }
  ])

  const [isTrading, setIsTrading] = useState(false)
  const [selectedTab, setSelectedTab] = useState('overview')

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Update market data
      setMarketData(prev => prev.map(stock => ({
        ...stock,
        price: stock.price * (1 + (Math.random() - 0.5) * 0.001),
        change: stock.change + (Math.random() - 0.5) * 0.1,
        change_percent: stock.change_percent + (Math.random() - 0.5) * 0.01,
        last_updated: new Date().toISOString()
      })))

      // Update positions
      setPositions(prev => prev.map(pos => {
        const marketStock = marketData.find(s => s.symbol === pos.symbol)
        if (marketStock) {
          const newPrice = marketStock.price
          const pnl = pos.type === 'LONG' 
            ? (newPrice - pos.entry_price) * pos.quantity
            : (pos.entry_price - newPrice) * pos.quantity
          const pnlPercent = (pnl / (pos.entry_price * pos.quantity)) * 100
          
          return {
            ...pos,
            current_price: newPrice,
            unrealized_pnl: pnl,
            unrealized_pnl_percent: pnlPercent,
            day_change: pnl - (pos.unrealized_pnl || 0),
            day_change_percent: pnlPercent - (pos.unrealized_pnl_percent || 0)
          }
        }
        return pos
      }))

      // Update portfolio
      setPortfolio(prev => {
        const totalUnrealizedPnl = positions.reduce((sum, pos) => sum + pos.unrealized_pnl, 0)
        const newTotalValue = prev.cash_balance + prev.invested_value + totalUnrealizedPnl
        const dayChange = newTotalValue - 125000 // Assuming previous close was 125000
        
        return {
          ...prev,
          total_value: newTotalValue,
          total_pnl: totalUnrealizedPnl,
          total_pnl_percent: (totalUnrealizedPnl / prev.invested_value) * 100,
          day_change: dayChange,
          day_change_percent: (dayChange / 125000) * 100
        }
      })
    }, 5000) // Update every 5 seconds

    return () => clearInterval(interval)
  }, [marketData, positions])

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value)
  }

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
  }

  const formatNumber = (value: number) => {
    return new Intl.NumberFormat('en-US').format(value)
  }

  const executeSignal = (signalId: string) => {
    setTradingSignals(prev => 
      prev.map(signal => 
        signal.id === signalId 
          ? { ...signal, executed: true }
          : signal
      )
    )
  }

  const toggleTrading = () => {
    setIsTrading(!isTrading)
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Trading Dashboard</h1>
          <p className="text-muted-foreground">Real-time trading overview and portfolio management</p>
        </div>
        <div className="flex items-center space-x-4">
          <Badge variant={isTrading ? "default" : "secondary"} className="flex items-center space-x-2">
            {isTrading ? <Activity className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
            <span>{isTrading ? 'Trading Active' : 'Trading Paused'}</span>
          </Badge>
          <Button
            onClick={toggleTrading}
            variant={isTrading ? "destructive" : "default"}
            className="flex items-center space-x-2"
          >
            {isTrading ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span>{isTrading ? 'Pause Trading' : 'Start Trading'}</span>
          </Button>
        </div>
      </div>

      {/* Portfolio Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Value</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatCurrency(portfolio.total_value)}</div>
            <p className="text-xs text-muted-foreground">
              {formatPercent(portfolio.day_change_percent)} today
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {formatCurrency(portfolio.total_pnl)}
            </div>
            <p className="text-xs text-muted-foreground">
              {formatPercent(portfolio.total_pnl_percent)} all time
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cash Balance</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatCurrency(portfolio.cash_balance)}</div>
            <p className="text-xs text-muted-foreground">
              {((portfolio.cash_balance / portfolio.total_value) * 100).toFixed(1)}% of portfolio
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Positions</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{positions.length}</div>
            <p className="text-xs text-muted-foreground">
              {positions.filter(p => p.unrealized_pnl > 0).length} profitable
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="positions">Positions</TabsTrigger>
          <TabsTrigger value="signals">Signals</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Active Positions */}
            <Card>
              <CardHeader>
                <CardTitle>Active Positions</CardTitle>
                <CardDescription>Current open positions and their performance</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {positions.map((position) => (
                    <div key={position.id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className={`w-2 h-2 rounded-full ${position.type === 'LONG' ? 'bg-green-500' : 'bg-red-500'}`} />
                        <div>
                          <p className="font-medium">{position.symbol}</p>
                          <p className="text-sm text-muted-foreground">
                            {position.quantity} shares @ {formatCurrency(position.entry_price)}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="font-medium">{formatCurrency(position.current_price)}</p>
                        <p className={`text-sm ${position.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {formatCurrency(position.unrealized_pnl)} ({formatPercent(position.unrealized_pnl_percent)})
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Market Overview */}
            <Card>
              <CardHeader>
                <CardTitle>Market Overview</CardTitle>
                <CardDescription>Real-time market data for watched symbols</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {marketData.map((stock) => (
                    <div key={stock.symbol} className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <p className="font-medium">{stock.symbol}</p>
                        <p className="text-sm text-muted-foreground">
                          Vol: {formatNumber(stock.volume)}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="font-medium">{formatCurrency(stock.price)}</p>
                        <p className={`text-sm ${stock.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)} ({formatPercent(stock.change_percent)})
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Trading Signals */}
          <Card>
            <CardHeader>
              <CardTitle>AI Trading Signals</CardTitle>
              <CardDescription>Latest trading signals from AI algorithms</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {tradingSignals.filter(signal => !signal.executed).map((signal) => (
                  <div key={signal.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <Badge variant={signal.action === 'BUY' ? 'default' : signal.action === 'SELL' ? 'destructive' : 'secondary'}>
                        {signal.action}
                      </Badge>
                      <div>
                        <p className="font-medium">{signal.symbol}</p>
                        <p className="text-sm text-muted-foreground">{signal.reason}</p>
                        <div className="flex items-center space-x-2 mt-1">
                          <span className="text-xs text-muted-foreground">Confidence:</span>
                          <Progress value={signal.confidence * 100} className="w-16 h-2" />
                          <span className="text-xs">{(signal.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>
                    <Button
                      size="sm"
                      onClick={() => executeSignal(signal.id)}
                      className="flex items-center space-x-1"
                    >
                      <CheckCircle className="w-4 h-4" />
                      <span>Execute</span>
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="positions" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Position Details</CardTitle>
              <CardDescription>Detailed view of all open positions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Symbol</th>
                      <th className="text-left p-2">Type</th>
                      <th className="text-left p-2">Quantity</th>
                      <th className="text-left p-2">Entry Price</th>
                      <th className="text-left p-2">Current Price</th>
                      <th className="text-left p-2">Unrealized P&L</th>
                      <th className="text-left p-2">Day Change</th>
                      <th className="text-left p-2">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((position) => (
                      <tr key={position.id} className="border-b">
                        <td className="p-2 font-medium">{position.symbol}</td>
                        <td className="p-2">
                          <Badge variant={position.type === 'LONG' ? 'default' : 'destructive'}>
                            {position.type}
                          </Badge>
                        </td>
                        <td className="p-2">{position.quantity}</td>
                        <td className="p-2">{formatCurrency(position.entry_price)}</td>
                        <td className="p-2">{formatCurrency(position.current_price)}</td>
                        <td className={`p-2 ${position.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {formatCurrency(position.unrealized_pnl)} ({formatPercent(position.unrealized_pnl_percent)})
                        </td>
                        <td className={`p-2 ${position.day_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {formatCurrency(position.day_change)} ({formatPercent(position.day_change_percent)})
                        </td>
                        <td className="p-2">
                          <div className="flex space-x-2">
                            <Button size="sm" variant="outline">Close</Button>
                            <Button size="sm" variant="outline">Adjust</Button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="signals" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Trading Signals History</CardTitle>
              <CardDescription>All trading signals generated by the AI system</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[...tradingSignals].reverse().map((signal) => (
                  <div key={signal.id} className={`p-4 border rounded-lg ${signal.executed ? 'bg-muted/50' : ''}`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <Badge variant={signal.action === 'BUY' ? 'default' : signal.action === 'SELL' ? 'destructive' : 'secondary'}>
                          {signal.action}
                        </Badge>
                        <div>
                          <p className="font-medium">{signal.symbol}</p>
                          <p className="text-sm text-muted-foreground">{signal.reason}</p>
                          <p className="text-xs text-muted-foreground mt-1">
                            {new Date(signal.timestamp).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <div className="text-right">
                          <div className="flex items-center space-x-2">
                            <span className="text-xs text-muted-foreground">Confidence:</span>
                            <Progress value={signal.confidence * 100} className="w-16 h-2" />
                            <span className="text-xs">{(signal.confidence * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                        {signal.executed && (
                          <Badge variant="outline" className="flex items-center space-x-1">
                            <CheckCircle className="w-3 h-3" />
                            <span>Executed</span>
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
                <CardDescription>Key performance indicators</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span>Win Rate</span>
                    <span className="font-medium">68.5%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Average Win</span>
                    <span className="font-medium text-green-600">$245.50</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Average Loss</span>
                    <span className="font-medium text-red-600">-$125.25</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Profit Factor</span>
                    <span className="font-medium">1.96</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Sharpe Ratio</span>
                    <span className="font-medium">1.24</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Max Drawdown</span>
                    <span className="font-medium text-red-600">-8.3%</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Risk Metrics</CardTitle>
                <CardDescription>Current risk assessment</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span>Portfolio Beta</span>
                    <span className="font-medium">0.85</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Value at Risk (95%)</span>
                    <span className="font-medium text-red-600">-$2,450</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Expected Shortfall</span>
                    <span className="font-medium text-red-600">-$3,200</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Position Concentration</span>
                    <span className="font-medium">32%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Sector Diversification</span>
                    <span className="font-medium">Good</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span>Liquidity Score</span>
                    <span className="font-medium text-green-600">High</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Risk Alerts</CardTitle>
              <CardDescription>Current risk warnings and recommendations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Moderate Risk</AlertTitle>
                  <AlertDescription>
                    Portfolio concentration in technology sector is above recommended levels. Consider diversifying.
                  </AlertDescription>
                </Alert>
                <Alert>
                  <TrendingUp className="h-4 w-4" />
                  <AlertTitle>Opportunity</AlertTitle>
                  <AlertDescription>
                    Market volatility is decreasing. Consider increasing position sizes in high-conviction trades.
                  </AlertDescription>
                </Alert>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}