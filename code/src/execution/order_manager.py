"""
Order Management System - AI Trading System

This module implements a comprehensive order management system that handles
order routing, execution, tracking, and optimization across multiple venues.

Educational Note:
Order Management Systems (OMS) are the backbone of modern trading.
They ensure orders are executed efficiently while managing risk,
monitoring compliance, and optimizing execution quality.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import asyncio
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import json
from collections import defaultdict

# Import our trading components
from .paper_trader import Order, OrderType, OrderSide, OrderStatus, Trade, MarketData
from .live_trader import LiveTradingEngine, BrokerConnector, ExecutionReport
from ..risk_management.position_sizer import PositionSizingManager
from ..risk_management.risk_calculator import RiskCalculator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderPriority(Enum):
    """Order execution priorities"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ExecutionVenue(Enum):
    """Execution venues"""
    PRIMARY_EXCHANGE = "primary_exchange"
    SECONDARY_EXCHANGE = "secondary_exchange"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    INTERNALIZER = "internalizer"
    RETAIL_MARKET_MAKER = "retail_market_maker"


class OrderRoutingStrategy(Enum):
    """Order routing strategies"""
    SIMPLE = "simple"
    SMART_ROUTING = "smart_routing"
    VENUE_OPTIMIZATION = "venue_optimization"
    COST_MINIMIZATION = "cost_minimization"
    SPEED_OPTIMIZATION = "speed_optimization"
    LIQUIDITY_SEEKING = "liquidity_seeking"


@dataclass
class OrderInstruction:
    """Detailed order instruction"""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    priority: OrderPriority = OrderPriority.NORMAL
    routing_strategy: OrderRoutingStrategy = OrderRoutingStrategy.SMART_ROUTING
    allowed_venues: List[ExecutionVenue] = field(default_factory=list)
    max_slippage: float = 0.001  # 0.1% max slippage
    execution_window: int = 300  # 5 minutes execution window
    min_fill_size: Optional[int] = None
    max_fill_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VenueInfo:
    """Information about execution venue"""
    venue: ExecutionVenue
    name: str
    fees: Dict[str, float]  # fee schedule
    liquidity_score: float  # 0-1, higher = more liquidity
    speed_score: float     # 0-1, higher = faster
    cost_score: float      # 0-1, higher = lower cost
    reliability_score: float  # 0-1, higher = more reliable
    supported_order_types: List[OrderType]
    min_order_size: int
    max_order_size: int
    average_spread: float
    market_hours: Dict[str, str]  # open/close times


@dataclass
class ExecutionPlan:
    """Plan for order execution"""
    order_id: str
    venue_allocations: Dict[ExecutionVenue, int]  # venue -> quantity
    routing_strategy: OrderRoutingStrategy
    estimated_cost: float
    estimated_time: timedelta
    risk_factors: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class OrderUpdate:
    """Order status update"""
    order_id: str
    status: OrderStatus
    filled_quantity: int
    avg_fill_price: float
    remaining_quantity: int
    timestamp: datetime
    venue: Optional[ExecutionVenue] = None
    notes: str = ""


class OrderRouter(ABC):
    """Abstract base class for order routers"""
    
    @abstractmethod
    async def route_order(self, instruction: OrderInstruction, venues: List[VenueInfo]) -> ExecutionPlan:
        """Create execution plan for order"""
        pass
    
    @abstractmethod
    async def execute_plan(self, plan: ExecutionPlan, instruction: OrderInstruction) -> List[ExecutionReport]:
        """Execute the order according to plan"""
        pass


class SimpleOrderRouter(OrderRouter):
    """
    Simple order router
    
    Educational Note:
    This is the most basic routing strategy that sends orders
    to a single venue based on simple rules.
    """
    
    def __init__(self):
        self.primary_venue = ExecutionVenue.PRIMARY_EXCHANGE
    
    async def route_order(self, instruction: OrderInstruction, venues: List[VenueInfo]) -> ExecutionPlan:
        """Route order to primary venue"""
        
        # Find best venue based on simple criteria
        best_venue = self._select_best_venue(instruction, venues)
        
        # Create execution plan
        plan = ExecutionPlan(
            order_id=str(uuid.uuid4()),
            venue_allocations={best_venue.venue: instruction.quantity},
            routing_strategy=OrderRoutingStrategy.SIMPLE,
            estimated_cost=self._estimate_cost(instruction, best_venue),
            estimated_time=timedelta(seconds=1),
            risk_factors={'liquidity_risk': 0.1, 'execution_risk': 0.05}
        )
        
        return plan
    
    async def execute_plan(self, plan: ExecutionPlan, instruction: OrderInstruction) -> List[ExecutionReport]:
        """Execute order on selected venue"""
        
        # This would integrate with the actual execution engine
        # For now, return a simulated execution report
        reports = []
        
        for venue, quantity in plan.venue_allocations.items():
            report = ExecutionReport(
                order_id=plan.order_id,
                symbol=instruction.symbol,
                side=instruction.side,
                quantity=quantity,
                filled_quantity=quantity,
                avg_price=instruction.price or 100.0,  # Simplified
                commission=quantity * 0.001,  # Simplified commission
                status=OrderStatus.FILLED,
                timestamp=datetime.now(),
                execution_venue=venue.value,
                liquidity="Available"
            )
            reports.append(report)
        
        return reports
    
    def _select_best_venue(self, instruction: OrderInstruction, venues: List[VenueInfo]) -> VenueInfo:
        """Select best venue based on simple criteria"""
        
        # Filter venues that support the order type
        suitable_venues = [v for v in venues if instruction.order_type in v.supported_order_types]
        
        if not suitable_venues:
            # Fallback to any venue
            suitable_venues = venues
        
        # Select venue with best combination of cost and liquidity
        best_venue = max(suitable_venues, key=lambda v: (v.liquidity_score + v.cost_score) / 2)
        
        return best_venue
    
    def _estimate_cost(self, instruction: OrderInstruction, venue: VenueInfo) -> float:
        """Estimate execution cost"""
        
        # Simplified cost estimation
        base_cost = instruction.quantity * venue.fees.get('per_share', 0.001)
        spread_cost = instruction.quantity * venue.average_spread * 0.5
        
        return base_cost + spread_cost


class SmartOrderRouter(OrderRouter):
    """
    Smart Order Router (SOR)
    
    Educational Note:
    Smart Order Routing uses real-time market data to find the best
    execution across multiple venues. It's used by institutional
    traders to minimize costs and improve execution quality.
    """
    
    def __init__(self):
        self.market_data_cache: Dict[str, MarketData] = {}
        self.venue_performance: Dict[ExecutionVenue, Dict] = defaultdict(lambda: defaultdict(list))
    
    async def route_order(self, instruction: OrderInstruction, venues: List[VenueInfo]) -> ExecutionPlan:
        """Create smart routing plan"""
        
        # Get real-time market data
        market_data = await self._get_market_data(instruction.symbol)
        
        # Analyze venue liquidity and pricing
        venue_analysis = self._analyze_venues(instruction, venues, market_data)
        
        # Optimize allocation across venues
        allocations = self._optimize_allocation(instruction, venue_analysis)
        
        # Calculate estimated cost and time
        estimated_cost = sum(
            self._calculate_venue_cost(venue, quantity, instruction)
            for venue, quantity in allocations.items()
        )
        
        estimated_time = self._estimate_execution_time(allocations, venue_analysis)
        
        # Assess risk factors
        risk_factors = self._assess_risk_factors(instruction, allocations, venue_analysis)
        
        plan = ExecutionPlan(
            order_id=str(uuid.uuid4()),
            venue_allocations=allocations,
            routing_strategy=OrderRoutingStrategy.SMART_ROUTING,
            estimated_cost=estimated_cost,
            estimated_time=estimated_time,
            risk_factors=risk_factors
        )
        
        return plan
    
    async def execute_plan(self, plan: ExecutionPlan, instruction: OrderInstruction) -> List[ExecutionReport]:
        """Execute smart routing plan"""
        
        reports = []
        
        # Execute orders in parallel across venues
        execution_tasks = []
        for venue, quantity in plan.venue_allocations.items():
            if quantity > 0:
                task = self._execute_on_venue(venue, quantity, instruction)
                execution_tasks.append(task)
        
        # Wait for all executions
        if execution_tasks:
            results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Execution error: {result}")
                else:
                    reports.extend(result)
        
        return reports
    
    async def _get_market_data(self, symbol: str) -> MarketData:
        """Get market data for symbol"""
        
        # This would integrate with real market data feeds
        # For now, return simulated data
        if symbol not in self.market_data_cache:
            self.market_data_cache[symbol] = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                bid=99.50,
                ask=100.50,
                bid_size=1000,
                ask_size=1000,
                last=100.00,
                volume=1000000,
                open=98.00,
                high=101.00,
                low=97.50,
                close=100.00
            )
        
        return self.market_data_cache[symbol]
    
    def _analyze_venues(self, instruction: OrderInstruction, venues: List[VenueInfo], market_data: MarketData) -> Dict[ExecutionVenue, Dict]:
        """Analyze venues for execution quality"""
        
        analysis = {}
        
        for venue in venues:
            # Calculate effective spread
            effective_spread = market_data.ask - market_data.bid
            
            # Estimate fill probability
            fill_probability = min(1.0, venue.liquidity_score * 0.8)
            
            # Calculate expected cost
            expected_cost = (
                effective_spread * 0.5 +  # Half spread
                venue.fees.get('per_share', 0.001)  # Commission
            )
            
            # Estimate execution time
            execution_time = timedelta(
                seconds=1.0 / venue.speed_score if venue.speed_score > 0 else 10
            )
            
            analysis[venue.venue] = {
                'effective_spread': effective_spread,
                'fill_probability': fill_probability,
                'expected_cost': expected_cost,
                'execution_time': execution_time,
                'available_liquidity': venue.liquidity_score * 10000  # Simplified
            }
        
        return analysis
    
    def _optimize_allocation(self, instruction: OrderInstruction, venue_analysis: Dict[ExecutionVenue, Dict]) -> Dict[ExecutionVenue, int]:
        """Optimize order allocation across venues"""
        
        remaining_quantity = instruction.quantity
        allocations = {}
        
        # Sort venues by cost (lowest first)
        sorted_venues = sorted(
            venue_analysis.items(),
            key=lambda x: x[1]['expected_cost']
        )
        
        for venue, analysis in sorted_venues:
            if remaining_quantity <= 0:
                break
            
            # Calculate optimal quantity for this venue
            available_liquidity = analysis['available_liquidity']
            optimal_quantity = min(remaining_quantity, available_liquidity)
            
            if optimal_quantity > 0:
                allocations[venue] = int(optimal_quantity)
                remaining_quantity -= int(optimal_quantity)
        
        # If we still have remaining quantity, allocate to best venue
        if remaining_quantity > 0 and sorted_venues:
            best_venue = sorted_venues[0][0]
            allocations[best_venue] = allocations.get(best_venue, 0) + remaining_quantity
        
        return allocations
    
    def _calculate_venue_cost(self, venue: ExecutionVenue, quantity: int, instruction: OrderInstruction) -> float:
        """Calculate execution cost for specific venue"""
        
        # This would use real venue data
        # Simplified calculation
        return quantity * 0.001  # 0.1% per share
    
    def _estimate_execution_time(self, allocations: Dict[ExecutionVenue, int], venue_analysis: Dict[ExecutionVenue, Dict]) -> timedelta:
        """Estimate total execution time"""
        
        max_time = timedelta(0)
        
        for venue, quantity in allocations.items():
            if venue in venue_analysis:
                venue_time = venue_analysis[venue]['execution_time']
                max_time = max(max_time, venue_time)
        
        return max_time
    
    def _assess_risk_factors(self, instruction: OrderInstruction, allocations: Dict[ExecutionVenue, int], venue_analysis: Dict[ExecutionVenue, Dict]) -> Dict[str, float]:
        """Assess execution risk factors"""
        
        # Liquidity risk
        total_liquidity = sum(
            venue_analysis.get(venue, {}).get('available_liquidity', 0)
            for venue in allocations.keys()
        )
        liquidity_risk = max(0, 1 - (instruction.quantity / total_liquidity)) if total_liquidity > 0 else 1.0
        
        # Execution risk
        avg_fill_probability = np.mean([
            venue_analysis.get(venue, {}).get('fill_probability', 0.5)
            for venue in allocations.keys()
        ])
        execution_risk = 1 - avg_fill_probability
        
        # Concentration risk
        concentration_risk = max(allocations.values()) / instruction.quantity if instruction.quantity > 0 else 0
        
        return {
            'liquidity_risk': liquidity_risk,
            'execution_risk': execution_risk,
            'concentration_risk': concentration_risk,
            'overall_risk': (liquidity_risk + execution_risk + concentration_risk) / 3
        }
    
    async def _execute_on_venue(self, venue: ExecutionVenue, quantity: int, instruction: OrderInstruction) -> List[ExecutionReport]:
        """Execute order on specific venue"""
        
        # This would integrate with the actual venue
        # For now, simulate execution
        await asyncio.sleep(0.1)  # Simulate execution delay
        
        report = ExecutionReport(
            order_id=str(uuid.uuid4()),
            symbol=instruction.symbol,
            side=instruction.side,
            quantity=quantity,
            filled_quantity=quantity,
            avg_price=instruction.price or 100.0,
            commission=quantity * 0.001,
            status=OrderStatus.FILLED,
            timestamp=datetime.now(),
            execution_venue=venue.value,
            liquidity="Available"
        )
        
        return [report]


class OrderManagementSystem:
    """
    Order Management System (OMS)
    
    Educational Note:
    The OMS is the central hub for all order-related activities.
    It handles order creation, routing, execution, tracking,
    and post-trade analysis while ensuring compliance and risk management.
    """
    
    def __init__(self, trading_engine: LiveTradingEngine):
        self.trading_engine = trading_engine
        self.orders: Dict[str, Order] = {}
        self.execution_plans: Dict[str, ExecutionPlan] = {}
        self.order_updates: List[OrderUpdate] = []
        
        # Order routing
        self.routers: Dict[OrderRoutingStrategy, OrderRouter] = {
            OrderRoutingStrategy.SIMPLE: SimpleOrderRouter(),
            OrderRoutingStrategy.SMART_ROUTING: SmartOrderRouter()
        }
        
        # Venue information
        self.venues: Dict[ExecutionVenue, VenueInfo] = {}
        self._initialize_venues()
        
        # Monitoring and analytics
        self.execution_stats: Dict[str, Any] = defaultdict(list)
        self.compliance_checks: List[Callable] = []
        
        # Event handlers
        self.order_handlers: List[Callable] = []
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_orders = 0
        self.successful_executions = 0
    
    def _initialize_venues(self):
        """Initialize venue information"""
        
        self.venues[ExecutionVenue.PRIMARY_EXCHANGE] = VenueInfo(
            venue=ExecutionVenue.PRIMARY_EXCHANGE,
            name="Primary Exchange",
            fees={'per_share': 0.001},
            liquidity_score=0.9,
            speed_score=0.8,
            cost_score=0.7,
            reliability_score=0.95,
            supported_order_types=[OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LOSS],
            min_order_size=1,
            max_order_size=1000000,
            average_spread=0.01,
            market_hours={'open': '09:30', 'close': '16:00'}
        )
        
        self.venues[ExecutionVenue.DARK_POOL] = VenueInfo(
            venue=ExecutionVenue.DARK_POOL,
            name="Dark Pool",
            fees={'per_share': 0.0005},
            liquidity_score=0.7,
            speed_score=0.6,
            cost_score=0.9,
            reliability_score=0.85,
            supported_order_types=[OrderType.LIMIT],
            min_order_size=100,
            max_order_size=500000,
            average_spread=0.005,
            market_hours={'open': '09:30', 'close': '16:00'}
        )
        
        self.venues[ExecutionVenue.ECN] = VenueInfo(
            venue=ExecutionVenue.ECN,
            name="ECN",
            fees={'per_share': 0.0008},
            liquidity_score=0.8,
            speed_score=0.9,
            cost_score=0.8,
            reliability_score=0.9,
            supported_order_types=[OrderType.MARKET, OrderType.LIMIT],
            min_order_size=1,
            max_order_size=100000,
            average_spread=0.008,
            market_hours={'open': '04:00', 'close': '20:00'}
        )
    
    async def submit_order(self, instruction: OrderInstruction) -> str:
        """Submit new order"""
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Create order object
        order = Order(
            id=order_id,
            symbol=instruction.symbol,
            side=instruction.side,
            order_type=instruction.order_type,
            quantity=instruction.quantity,
            price=instruction.price,
            stop_price=instruction.stop_price,
            time_in_force=instruction.time_in_force,
            metadata=instruction.metadata.copy()
        )
        
        # Store order
        self.orders[order_id] = order
        self.total_orders += 1
        
        # Pre-execution checks
        if not await self._pre_execution_checks(order, instruction):
            order.status = OrderStatus.REJECTED
            await self._notify_order_update(order_id, OrderStatus.REJECTED, 0, 0.0, instruction.quantity)
            return order_id
        
        # Create execution plan
        router = self.routers.get(instruction.routing_strategy, self.routers[OrderRoutingStrategy.SIMPLE])
        allowed_venues = instruction.allowed_venues or list(self.venues.keys())
        venue_list = [self.venues[v] for v in allowed_venues if v in self.venues]
        
        try:
            plan = await router.route_order(instruction, venue_list)
            self.execution_plans[order_id] = plan
            
            # Execute order
            reports = await router.execute_plan(plan, instruction)
            
            # Process execution results
            await self._process_execution_results(order_id, reports)
            
            self.successful_executions += 1
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            order.status = OrderStatus.REJECTED
            await self._notify_order_update(order_id, OrderStatus.REJECTED, 0, 0.0, instruction.quantity)
        
        return order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        
        order = self.orders.get(order_id)
        if not order:
            return False
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        try:
            # Cancel through trading engine
            success = await self.trading_engine.cancel_order(order_id)
            
            if success:
                order.status = OrderStatus.CANCELLED
                await self._notify_order_update(order_id, OrderStatus.CANCELLED, 
                                             order.filled_quantity, order.avg_fill_price, 0)
            
            return success
            
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False
    
    async def modify_order(self, order_id: str, new_instruction: OrderInstruction) -> bool:
        """Modify existing order"""
        
        order = self.orders.get(order_id)
        if not order:
            return False
        
        if order.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]:
            return False
        
        # Cancel original order
        cancel_success = await self.cancel_order(order_id)
        
        if cancel_success:
            # Submit new order
            await self.submit_order(new_instruction)
            return True
        
        return False
    
    async def _pre_execution_checks(self, order: Order, instruction: OrderInstruction) -> bool:
        """Perform pre-execution checks"""
        
        # Risk management checks
        if not await self._risk_check(order, instruction):
            return False
        
        # Compliance checks
        if not await self._compliance_check(order, instruction):
            return False
        
        # Venue availability checks
        if not await self._venue_check(instruction):
            return False
        
        return True
    
    async def _risk_check(self, order: Order, instruction: OrderInstruction) -> bool:
        """Check order against risk limits"""
        
        # This would integrate with risk management system
        # For now, basic checks
        
        # Check order size
        if order.quantity > 1000000:  # Simplified limit
            logger.warning(f"Order size {order.quantity} exceeds limit")
            return False
        
        return True
    
    async def _compliance_check(self, order: Order, instruction: OrderInstruction) -> bool:
        """Check order compliance"""
        
        for check in self.compliance_checks:
            if not await check(order, instruction):
                return False
        
        return True
    
    async def _venue_check(self, instruction: OrderInstruction) -> bool:
        """Check venue availability"""
        
        if instruction.allowed_venues:
            for venue in instruction.allowed_venues:
                if venue in self.venues:
                    # Check if venue is open
                    venue_info = self.venues[venue]
                    if self._is_venue_open(venue_info):
                        return True
        
        return True
    
    def _is_venue_open(self, venue_info: VenueInfo) -> bool:
        """Check if venue is open"""
        
        # Simplified check - in reality, this would consider holidays, etc.
        now = datetime.now().time()
        
        try:
            open_time = datetime.strptime(venue_info.market_hours['open'], '%H:%M').time()
            close_time = datetime.strptime(venue_info.market_hours['close'], '%H:%M').time()
            
            return open_time <= now <= close_time
            
        except:
            return True  # Default to open
    
    async def _process_execution_results(self, order_id: str, reports: List[ExecutionReport]):
        """Process execution results"""
        
        order = self.orders.get(order_id)
        if not order:
            return
        
        total_filled = 0
        total_cost = 0.0
        
        for report in reports:
            total_filled += report.filled_quantity
            total_cost += report.filled_quantity * report.avg_fill_price + report.commission
        
        # Update order
        order.filled_quantity = total_filled
        order.avg_fill_price = total_cost / total_filled if total_filled > 0 else 0.0
        order.commission += sum(r.commission for r in reports)
        
        if total_filled >= order.quantity:
            order.status = OrderStatus.FILLED
        elif total_filled > 0:
            order.status = OrderStatus.PARTIAL_FILLED
        
        # Record execution statistics
        self._record_execution_stats(order, reports)
        
        # Notify handlers
        await self._notify_order_update(order_id, order.status, total_filled, 
                                      order.avg_fill_price, order.quantity - total_filled)
    
    def _record_execution_stats(self, order: Order, reports: List[ExecutionReport]):
        """Record execution statistics"""
        
        for report in reports:
            # Calculate slippage
            expected_price = order.price or 0.0
            slippage = (report.avg_fill_price - expected_price) / expected_price if expected_price > 0 else 0.0
            
            # Record stats
            self.execution_stats['slippage'].append(slippage)
            self.execution_stats['execution_time'].append(
                (report.timestamp - order.created_at).total_seconds()
            )
            self.execution_stats['fill_ratio'].append(
                report.filled_quantity / order.quantity if order.quantity > 0 else 0
            )
    
    async def _notify_order_update(self, order_id: str, status: OrderStatus, 
                                 filled_qty: int, avg_price: float, remaining_qty: int):
        """Notify order update to handlers"""
        
        update = OrderUpdate(
            order_id=order_id,
            status=status,
            filled_quantity=filled_qty,
            avg_fill_price=avg_price,
            remaining_quantity=remaining_qty,
            timestamp=datetime.now()
        )
        
        self.order_updates.append(update)
        
        # Notify handlers
        for handler in self.order_handlers:
            try:
                await handler(update)
            except Exception as e:
                logger.error(f"Error in order handler: {e}")
    
    def add_order_handler(self, handler: Callable):
        """Add order update handler"""
        self.order_handlers.append(handler)
    
    def add_compliance_check(self, check: Callable):
        """Add compliance check"""
        self.compliance_checks.append(check)
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status"""
        order = self.orders.get(order_id)
        return order.status if order else None
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details"""
        return self.orders.get(order_id)
    
    def get_execution_plan(self, order_id: str) -> Optional[ExecutionPlan]:
        """Get execution plan"""
        return self.execution_plans.get(order_id)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get OMS performance metrics"""
        
        uptime = datetime.now() - self.start_time
        
        # Calculate success rate
        success_rate = self.successful_executions / self.total_orders if self.total_orders > 0 else 0
        
        # Calculate average metrics
        avg_slippage = np.mean(self.execution_stats['slippage']) if self.execution_stats['slippage'] else 0
        avg_execution_time = np.mean(self.execution_stats['execution_time']) if self.execution_stats['execution_time'] else 0
        avg_fill_ratio = np.mean(self.execution_stats['fill_ratio']) if self.execution_stats['fill_ratio'] else 0
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'total_orders': self.total_orders,
            'successful_executions': self.successful_executions,
            'success_rate': success_rate,
            'average_slippage': avg_slippage,
            'average_execution_time': avg_execution_time,
            'average_fill_ratio': avg_fill_ratio,
            'active_orders': len([o for o in self.orders.values() if o.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]]),
            'venues_count': len(self.venues),
            'routing_strategies': list(self.routers.keys())
        }
    
    def get_venue_performance(self) -> Dict[str, Dict]:
        """Get venue performance statistics"""
        
        performance = {}
        
        for venue in self.venues.keys():
            # This would track actual performance by venue
            performance[venue.value] = {
                'orders_executed': 0,
                'total_volume': 0,
                'average_slippage': 0.0,
                'fill_rate': 0.0,
                'reliability': self.venues[venue].reliability_score
            }
        
        return performance


def explain_order_management():
    """
    Educational explanation of order management systems
    """
    
    print("=== Order Management System Educational Guide ===\n")
    
    concepts = {
        'Order Management System (OMS)': "Central system that handles the entire order lifecycle from creation to execution",
        
        'Smart Order Routing (SOR)': "Algorithm that finds the best execution across multiple trading venues",
        
        'Execution Venue': "Place where orders are executed (exchanges, dark pools, ECNs, etc.)",
        
        'Liquidity': "Ability to execute orders without significantly affecting price",
        
        'Slippage': "Difference between expected and actual execution price",
        
        'Fill Ratio': "Percentage of order that gets executed",
        
        'Order Routing Strategy': "Method used to decide where and how to send orders",
        
        'Compliance Checking': "Ensuring orders meet regulatory and internal requirements",
        
        'Execution Quality': "Measure of how well orders were executed (cost, speed, fill rate)"
    }
    
    for concept, explanation in concepts.items():
        print(f"{concept}:")
        print(f"  {explanation}\n")
    
    print("=== Order Routing Strategies ===")
    strategies = {
        'Simple Routing': "Send all orders to a single pre-selected venue",
        'Smart Routing': "Dynamically route to best venue based on real-time conditions",
        'Venue Optimization': "Allocate orders across multiple venues to minimize cost",
        'Liquidity Seeking': "Find hidden liquidity across multiple venues",
        'Speed Optimization': "Prioritize fastest execution over cost"
    }
    
    for strategy, description in strategies.items():
        print(f"{strategy}:")
        print(f"  {description}\n")
    
    print("=== Execution Venues ===")
    venues = {
        'Primary Exchange': "Main stock exchange (NYSE, NASDAQ) with high liquidity and transparency",
        'Dark Pool': 'Private trading venues with non-transparent pricing',
        'ECN': 'Electronic Communication Networks that match buyers and sellers',
        'Internalizer': 'Brokerage firms that execute orders internally',
        'Retail Market Makers': 'Firms that specialize in retail order flow'
    }
    
    for venue, description in venues.items():
        print(f"{venue}:")
        print(f"  {description}\n")
    
    print("=== OMS Best Practices ===")
    practices = [
        "1. Use smart order routing to minimize costs",
        "2. Monitor execution quality in real-time",
        "3. Implement comprehensive compliance checks",
        "4. Track venue performance and adapt routing",
        "5. Handle partial fills and order modifications",
        "6. Maintain detailed audit trails",
        "7. Implement proper error handling",
        "8. Monitor system latency and performance",
        "9. Regular backtesting of routing strategies",
        "10. Maintain redundancy and failover systems"
    ]
    
    for practice in practices:
        print(practice)


if __name__ == "__main__":
    # Example usage
    explain_order_management()
    
    print("\n=== Order Management System Example ===")
    
    # This would require a live trading engine
    print("To use the OMS:")
    print("1. Create or connect to a live trading engine")
    print("2. Configure venues and routing strategies")
    print("3. Set up compliance checks")
    print("4. Add order update handlers")
    print("5. Submit orders and monitor execution")
    
    # Example of creating an order instruction
    instruction = OrderInstruction(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        price=150.00,
        routing_strategy=OrderRoutingStrategy.SMART_ROUTING,
        priority=OrderPriority.NORMAL
    )
    
    print(f"\nSample order instruction:")
    print(f"  Symbol: {instruction.symbol}")
    print(f"  Side: {instruction.side.value}")
    print(f"  Quantity: {instruction.quantity}")
    print(f"  Type: {instruction.order_type.value}")
    print(f"  Price: ${instruction.price}")
    print(f"  Routing: {instruction.routing_strategy.value}")
    
    print("\nThe OMS would:")
    print("1. Perform risk and compliance checks")
    print("2. Create smart routing plan")
    print("3. Execute across optimal venues")
    print("4. Track and report execution results")
    print("5. Update order status in real-time")