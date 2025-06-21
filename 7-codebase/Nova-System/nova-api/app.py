#!/usr/bin/env python3
"""
nova-api/src/api/app.py

Nova API - 全息现实融合系统REST/WebSocket API
提供场景管理、实时通信、监控等企业级API服务

技术栈：
- FastAPI: 高性能异步Web框架
- WebSocket: 实时双向通信
- Prometheus: 监控指标
- Redis: 分布式缓存（可选）

Novus Technology Co., Ltd. 深圳新生科技股份有限公司 (c) 2030

"""

import asyncio
import time
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from contextlib import asynccontextmanager
import os

from fastapi import (
    FastAPI, 
    HTTPException, 
    Depends, 
    WebSocket, 
    WebSocketDisconnect,
    Request,
    status,
    BackgroundTasks,
    Query
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

# 监控
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info,
        generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    
# Redis支持
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Nova核心引擎
from nova_core.core.engine import (
    NovaEngine, 
    Scene, 
    ExperienceLevel,
    SceneType,
    UserProfile
)

# ===== 配置 =====

class Settings:
    """应用配置"""
    APP_NAME = "Nova API"
    APP_VERSION = "3.0.0"
    
    # 环境
    ENVIRONMENT = os.getenv("NOVA_ENV", "development")
    DEBUG = ENVIRONMENT == "development"
    
    # QCCIF连接
    QCCIF_URL = os.getenv("QCCIF_URL", "http://localhost:8080")
    QCCIF_TOKEN = os.getenv("QCCIF_TOKEN", "")
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # 安全
    SECRET_KEY = os.getenv("SECRET_KEY", "nova-secret-key-change-in-production")
    API_KEY_HEADER = "X-API-Key"
    
    # 限制
    MAX_SCENES_PER_USER = 10
    MAX_WEBSOCKET_CONNECTIONS = 10000
    DEFAULT_SCENE_TIMEOUT = 3600  # 1小时
    
    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # 日志
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()

# ===== 日志配置 =====

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== 监控指标 =====

if PROMETHEUS_AVAILABLE:
    # 请求计数器
    request_count = Counter(
        'nova_api_requests_total',
        'Total number of API requests',
        ['method', 'endpoint', 'status']
    )
    
    # 请求延迟
    request_duration = Histogram(
        'nova_api_request_duration_seconds',
        'Request duration in seconds',
        ['method', 'endpoint'],
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
    )
    
    # 场景指标
    scene_creation_counter = Counter(
        'nova_scenes_created_total',
        'Total number of scenes created',
        ['scene_type', 'experience_level']
    )
    
    scene_creation_duration = Histogram(
        'nova_scene_creation_duration_seconds',
        'Scene creation duration',
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
    )
    
    active_scenes_gauge = Gauge(
        'nova_active_scenes',
        'Number of active scenes',
        ['scene_type']
    )
    
    websocket_connections_gauge = Gauge(
        'nova_websocket_connections',
        'Number of active WebSocket connections'
    )
    
    # 缓存指标
    cache_hit_counter = Counter(
        'nova_cache_hits_total',
        'Total cache hits'
    )
    
    cache_miss_counter = Counter(
        'nova_cache_misses_total',
        'Total cache misses'
    )
    
    # 错误计数
    error_counter = Counter(
        'nova_api_errors_total',
        'Total number of errors',
        ['error_type']
    )

# ===== 数据模型 =====

class CreateSceneRequest(BaseModel):
    """创建场景请求"""
    scene_type: str = Field(..., description="场景类型")
    parameters: Dict[str, Any] = Field(..., description="场景参数")
    budget_limit: Optional[float] = Field(None, description="预算限制")
    priority: str = Field("normal", description="优先级: low/normal/high")
    
    @validator('scene_type')
    def validate_scene_type(cls, v):
        try:
            SceneType(v)
        except ValueError:
            valid_types = [t.value for t in SceneType]
            raise ValueError(f"Invalid scene type. Must be one of: {valid_types}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "scene_type": "pet_companion",
                "parameters": {
                    "pet_name": "小花",
                    "species": "cat",
                    "personality": "gentle",
                    "quality": "auto"
                },
                "budget_limit": 50.0,
                "priority": "normal"
            }
        }

class SceneResponse(BaseModel):
    """场景响应"""
    scene_id: str
    status: str
    scene_type: str
    experience_level: str
    estimated_cost: float
    actual_cost: float
    cache_hit: bool
    quantum_job_id: Optional[str]
    features: List[str]
    websocket_url: str
    created_at: datetime
    
class SceneUpdateRequest(BaseModel):
    """场景更新请求"""
    updates: Dict[str, Any]
    
class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    
class AlternativeOption(BaseModel):
    """替代方案"""
    level: str
    features: List[str]
    estimated_cost: float
    savings: float
    
class BudgetExceededResponse(BaseModel):
    """预算超出响应"""
    error: str = "budget_exceeded"
    message: str
    available_budget: float
    requested_cost: float
    alternatives: List[AlternativeOption]
    
class SceneMetrics(BaseModel):
    """场景指标"""
    scene_id: str
    uptime_seconds: float
    quantum_stability: float
    user_satisfaction: float
    resource_usage: Dict[str, Any]
    
class UserInfo(BaseModel):
    """用户信息"""
    user_id: str
    tier: str
    daily_budget: float
    used_budget: float
    active_scenes: int
    
# ===== 认证 =====

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """获取当前用户ID（简化版本）"""
    # TODO: 实现真实的JWT验证
    # 这里简单地从token中提取用户ID
    token = credentials.credentials
    
    # 模拟token解析
    if token.startswith("user_"):
        return token
    else:
        # 默认测试用户
        return "user_test_001"
        
# ===== 依赖注入 =====

def get_nova_engine(request: Request) -> NovaEngine:
    """获取Nova引擎实例"""
    return request.app.state.nova_engine

def get_redis_client(request: Request) -> Optional[redis.Redis]:
    """获取Redis客户端"""
    return getattr(request.app.state, 'redis_client', None)

# ===== WebSocket管理 =====

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # scene_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # WebSocket -> user_id
        self.connection_users: Dict[WebSocket, str] = {}
        self.logger = logging.getLogger(__name__)
        
    async def connect(
        self, 
        websocket: WebSocket, 
        scene_id: str, 
        user_id: str
    ):
        """建立连接"""
        await websocket.accept()
        
        if scene_id not in self.active_connections:
            self.active_connections[scene_id] = set()
            
        self.active_connections[scene_id].add(websocket)
        self.connection_users[websocket] = user_id
        
        if PROMETHEUS_AVAILABLE:
            websocket_connections_gauge.inc()
            
        self.logger.info(
            f"WebSocket connected: user={user_id}, scene={scene_id}"
        )
        
    def disconnect(self, websocket: WebSocket, scene_id: str):
        """断开连接"""
        if scene_id in self.active_connections:
            self.active_connections[scene_id].discard(websocket)
            
            if not self.active_connections[scene_id]:
                del self.active_connections[scene_id]
                
        if websocket in self.connection_users:
            del self.connection_users[websocket]
            
        if PROMETHEUS_AVAILABLE:
            websocket_connections_gauge.dec()
            
    async def send_personal_message(
        self, 
        message: Dict[str, Any], 
        websocket: WebSocket
    ):
        """发送个人消息"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            
    async def broadcast_to_scene(
        self, 
        scene_id: str, 
        message: Dict[str, Any]
    ):
        """广播到场景的所有连接"""
        if scene_id not in self.active_connections:
            return
            
        # 添加时间戳
        message['timestamp'] = datetime.now().isoformat()
        
        # 并发发送到所有连接
        tasks = []
        for connection in self.active_connections[scene_id]:
            tasks.append(self.send_personal_message(message, connection))
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
    def get_scene_connections(self, scene_id: str) -> int:
        """获取场景的连接数"""
        return len(self.active_connections.get(scene_id, set()))
        
    def has_connections(self, scene_id: str) -> bool:
        """检查场景是否有活跃连接"""
        return scene_id in self.active_connections and \
               len(self.active_connections[scene_id]) > 0

# ===== 错误处理 =====

class NovaException(Exception):
    """Nova基础异常"""
    def __init__(self, message: str, status_code: int = 500, details: Any = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)
        
class BudgetExceededException(NovaException):
    """预算超出异常"""
    def __init__(self, message: str, available: float, requested: float):
        super().__init__(message, status_code=402)
        self.available = available
        self.requested = requested
        
class SceneNotFoundException(NovaException):
    """场景未找到异常"""
    def __init__(self, scene_id: str):
        super().__init__(f"Scene {scene_id} not found", status_code=404)
        
class QuotaExceededException(NovaException):
    """配额超出异常"""
    def __init__(self, message: str):
        super().__init__(message, status_code=429)

# ===== 熔断器 =====

class CircuitBreaker:
    """熔断器模式实现"""
    
    def __init__(
        self, 
        failure_threshold: int = 5, 
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
        
    async def call(self, func, *args, **kwargs):
        """通过熔断器调用函数"""
        # 检查是否应该尝试恢复
        if self.is_open:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.is_open = False
                self.failure_count = 0
            else:
                raise NovaException(
                    "Service temporarily unavailable",
                    status_code=503
                )
                
        try:
            result = await func(*args, **kwargs)
            # 成功调用，重置失败计数
            self.failure_count = 0
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
                
            raise e

# ===== 应用生命周期 =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting Nova API...")
    
    # 初始化Nova引擎
    app.state.nova_engine = NovaEngine()
    await app.state.nova_engine.initialize()
    logger.info("Nova Engine initialized")
    
    # 初始化Redis（如果可用）
    if REDIS_AVAILABLE and settings.REDIS_URL:
        try:
            app.state.redis_client = await redis.from_url(settings.REDIS_URL)
            await app.state.redis_client.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            app.state.redis_client = None
    
    # 初始化WebSocket管理器
    app.state.connection_manager = ConnectionManager()
    
    # 初始化熔断器
    app.state.circuit_breaker = CircuitBreaker()
    
    logger.info("Nova API started successfully")
    
    yield
    
    # 清理
    logger.info("Shutting down Nova API...")
    
    # 关闭Nova引擎
    await app.state.nova_engine.shutdown()
    
    # 关闭Redis连接
    if hasattr(app.state, 'redis_client') and app.state.redis_client:
        await app.state.redis_client.close()
        
    logger.info("Nova API shutdown complete")

# ===== 创建应用 =====

app = FastAPI(
    title=settings.APP_NAME,
    description="全息现实融合系统API - 让虚拟比真实更真实",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# ===== 中间件 =====

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip压缩
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 请求计时中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加处理时间头"""
    start_time = time.time()
    
    # 记录请求
    if PROMETHEUS_AVAILABLE:
        method = request.method
        path = request.url.path
        
    response = await call_next(request)
    
    # 计算处理时间
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # 更新指标
    if PROMETHEUS_AVAILABLE:
        request_count.labels(
            method=method,
            endpoint=path,
            status=response.status_code
        ).inc()
        
        request_duration.labels(
            method=method,
            endpoint=path
        ).observe(process_time)
        
    return response

# ===== 异常处理器 =====

@app.exception_handler(NovaException)
async def nova_exception_handler(request: Request, exc: NovaException):
    """处理Nova自定义异常"""
    if PROMETHEUS_AVAILABLE:
        error_counter.labels(error_type=exc.__class__.__name__).inc()
        
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details
        }
    )

@app.exception_handler(BudgetExceededException)
async def budget_exceeded_handler(request: Request, exc: BudgetExceededException):
    """处理预算超出异常"""
    # 获取替代方案
    nova_engine = request.app.state.nova_engine
    
    # 从请求中提取参数（这里简化处理）
    alternatives = []
    
    return JSONResponse(
        status_code=402,
        content={
            "error": "budget_exceeded",
            "message": exc.message,
            "available_budget": exc.available,
            "requested_cost": exc.requested,
            "alternatives": alternatives
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """处理值错误"""
    return JSONResponse(
        status_code=400,
        content={
            "error": "validation_error",
            "message": str(exc)
        }
    )

# ===== API路由 =====

@app.get("/", tags=["Root"])
async def root():
    """API根路径"""
    return {
        "service": "Nova API",
        "version": settings.APP_VERSION,
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check(nova_engine: NovaEngine = Depends(get_nova_engine)):
    """健康检查"""
    try:
        # 检查QCCIF连接
        qccif_status = "connected" if nova_engine.qccif_engine else "simulation"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "api": "operational",
                "nova_engine": "operational",
                "qccif": qccif_status,
                "redis": "connected" if app.state.redis_client else "not_connected"
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus指标端点"""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Prometheus metrics not available"
        )
        
    # 生成指标
    metrics_data = generate_latest()
    
    return StreamingResponse(
        iter([metrics_data]),
        media_type=CONTENT_TYPE_LATEST
    )

# ===== 场景管理API =====

@app.post(
    "/api/v1/scenes",
    response_model=SceneResponse,
    tags=["Scenes"],
    summary="创建新场景",
    description="创建全息现实融合场景，支持智能预算优化"
)
async def create_scene(
    request: CreateSceneRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
    nova_engine: NovaEngine = Depends(get_nova_engine)
):
    """
    创建新场景
    
    - **scene_type**: 场景类型（pet_companion, virtual_meeting等）
    - **parameters**: 场景参数（根据类型不同而不同）
    - **budget_limit**: 预算限制（可选，不填则使用账户默认限额）
    - **priority**: 优先级（low/normal/high）
    
    系统会根据预算自动选择最佳体验等级。
    """
    start_time = time.time()
    
    try:
        # 检查用户场景数量限制
        user_scenes = await nova_engine.get_user_scenes(user_id)
        active_scenes = [s for s in user_scenes if s.status == 'active']
        
        if len(active_scenes) >= settings.MAX_SCENES_PER_USER:
            raise QuotaExceededException(
                f"Maximum {settings.MAX_SCENES_PER_USER} active scenes allowed"
            )
            
        # 创建场景
        scene = await nova_engine.create_scene(
            user_id=user_id,
            scene_type=request.scene_type,
            parameters=request.parameters,
            budget_limit=request.budget_limit
        )
        
        # 更新指标
        if PROMETHEUS_AVAILABLE:
            creation_time = time.time() - start_time
            scene_creation_counter.labels(
                scene_type=scene.scene_type.value,
                experience_level=scene.experience_level.value
            ).inc()
            scene_creation_duration.observe(creation_time)
            
            # 更新活跃场景数
            active_scenes_gauge.labels(
                scene_type=scene.scene_type.value
            ).inc()
            
            # 缓存指标
            if scene.cache_hit:
                cache_hit_counter.inc()
            else:
                cache_miss_counter.inc()
                
        # 后台任务：通知其他服务
        background_tasks.add_task(
            notify_scene_created,
            scene.id,
            user_id
        )
        
        # 返回响应
        return SceneResponse(
            scene_id=scene.id,
            status=scene.status,
            scene_type=scene.scene_type.value,
            experience_level=scene.experience_level.value,
            estimated_cost=scene.estimated_cost,
            actual_cost=scene.actual_cost,
            cache_hit=scene.cache_hit,
            quantum_job_id=scene.quantum_job_id,
            features=nova_engine._get_level_features(scene.experience_level),
            websocket_url=f"/ws/scenes/{scene.id}",
            created_at=scene.created_at
        )
        
    except BudgetExceededException as e:
        # 获取替代方案
        alternatives = await nova_engine.suggest_alternatives(
            request.scene_type,
            request.parameters,
            e.available
        )
        
        return JSONResponse(
            status_code=402,
            content=BudgetExceededResponse(
                message=e.message,
                available_budget=e.available,
                requested_cost=e.requested,
                alternatives=[
                    AlternativeOption(**alt) for alt in alternatives
                ]
            ).dict()
        )

@app.get(
    "/api/v1/scenes/{scene_id}",
    response_model=SceneResponse,
    tags=["Scenes"],
    summary="获取场景信息"
)
async def get_scene(
    scene_id: str,
    user_id: str = Depends(get_current_user),
    nova_engine: NovaEngine = Depends(get_nova_engine)
):
    """获取场景详细信息"""
    scene = await nova_engine.get_scene(scene_id)
    
    if not scene:
        raise SceneNotFoundException(scene_id)
        
    # 检查权限
    if scene.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
        
    return SceneResponse(
        scene_id=scene.id,
        status=scene.status,
        scene_type=scene.scene_type.value,
        experience_level=scene.experience_level.value,
        estimated_cost=scene.estimated_cost,
        actual_cost=scene.actual_cost,
        cache_hit=scene.cache_hit,
        quantum_job_id=scene.quantum_job_id,
        features=nova_engine._get_level_features(scene.experience_level),
        websocket_url=f"/ws/scenes/{scene.id}",
        created_at=scene.created_at
    )

@app.get(
    "/api/v1/scenes",
    response_model=List[SceneResponse],
    tags=["Scenes"],
    summary="列出用户场景"
)
async def list_user_scenes(
    user_id: str = Depends(get_current_user),
    nova_engine: NovaEngine = Depends(get_nova_engine),
    status: Optional[str] = Query(None, description="过滤场景状态"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """列出用户的所有场景"""
    scenes = await nova_engine.get_user_scenes(user_id)
    
    # 过滤状态
    if status:
        scenes = [s for s in scenes if s.status == status]
        
    # 分页
    total = len(scenes)
    scenes = scenes[offset:offset + limit]
    
    # 转换响应
    responses = []
    for scene in scenes:
        responses.append(SceneResponse(
            scene_id=scene.id,
            status=scene.status,
            scene_type=scene.scene_type.value,
            experience_level=scene.experience_level.value,
            estimated_cost=scene.estimated_cost,
            actual_cost=scene.actual_cost,
            cache_hit=scene.cache_hit,
            quantum_job_id=scene.quantum_job_id,
            features=nova_engine._get_level_features(scene.experience_level),
            websocket_url=f"/ws/scenes/{scene.id}",
            created_at=scene.created_at
        ))
        
    return responses

@app.patch(
    "/api/v1/scenes/{scene_id}",
    response_model=SceneResponse,
    tags=["Scenes"],
    summary="更新场景"
)
async def update_scene(
    scene_id: str,
    request: SceneUpdateRequest,
    user_id: str = Depends(get_current_user),
    nova_engine: NovaEngine = Depends(get_nova_engine)
):
    """更新场景参数"""
    # 检查场景存在性和权限
    scene = await nova_engine.get_scene(scene_id)
    if not scene:
        raise SceneNotFoundException(scene_id)
        
    if scene.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
        
    # 更新场景
    updated_scene = await nova_engine.update_scene(
        scene_id,
        request.updates
    )
    
    return SceneResponse(
        scene_id=updated_scene.id,
        status=updated_scene.status,
        scene_type=updated_scene.scene_type.value,
        experience_level=updated_scene.experience_level.value,
        estimated_cost=updated_scene.estimated_cost,
        actual_cost=updated_scene.actual_cost,
        cache_hit=updated_scene.cache_hit,
        quantum_job_id=updated_scene.quantum_job_id,
        features=nova_engine._get_level_features(updated_scene.experience_level),
        websocket_url=f"/ws/scenes/{updated_scene.id}",
        created_at=updated_scene.created_at
    )

@app.post(
    "/api/v1/scenes/{scene_id}/pause",
    tags=["Scenes"],
    summary="暂停场景"
)
async def pause_scene(
    scene_id: str,
    user_id: str = Depends(get_current_user),
    nova_engine: NovaEngine = Depends(get_nova_engine)
):
    """暂停场景执行"""
    # 检查权限
    scene = await nova_engine.get_scene(scene_id)
    if not scene:
        raise SceneNotFoundException(scene_id)
        
    if scene.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
        
    await nova_engine.pause_scene(scene_id)
    
    return {"message": "Scene paused successfully"}

@app.post(
    "/api/v1/scenes/{scene_id}/resume",
    tags=["Scenes"],
    summary="恢复场景"
)
async def resume_scene(
    scene_id: str,
    user_id: str = Depends(get_current_user),
    nova_engine: NovaEngine = Depends(get_nova_engine)
):
    """恢复暂停的场景"""
    # 检查权限
    scene = await nova_engine.get_scene(scene_id)
    if not scene:
        raise SceneNotFoundException(scene_id)
        
    if scene.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
        
    await nova_engine.resume_scene(scene_id)
    
    return {"message": "Scene resumed successfully"}

@app.delete(
    "/api/v1/scenes/{scene_id}",
    tags=["Scenes"],
    summary="终止场景"
)
async def terminate_scene(
    scene_id: str,
    user_id: str = Depends(get_current_user),
    nova_engine: NovaEngine = Depends(get_nova_engine)
):
    """终止并删除场景"""
    # 检查权限
    scene = await nova_engine.get_scene(scene_id)
    if not scene:
        raise SceneNotFoundException(scene_id)
        
    if scene.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
        
    await nova_engine.terminate_scene(scene_id)
    
    # 更新指标
    if PROMETHEUS_AVAILABLE:
        active_scenes_gauge.labels(
            scene_type=scene.scene_type.value
        ).dec()
        
    return {"message": "Scene terminated successfully"}

@app.get(
    "/api/v1/scenes/{scene_id}/metrics",
    response_model=SceneMetrics,
    tags=["Scenes"],
    summary="获取场景指标"
)
async def get_scene_metrics(
    scene_id: str,
    user_id: str = Depends(get_current_user),
    nova_engine: NovaEngine = Depends(get_nova_engine)
):
    """获取场景运行指标"""
    # 检查权限
    scene = await nova_engine.get_scene(scene_id)
    if not scene:
        raise SceneNotFoundException(scene_id)
        
    if scene.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
        
    metrics = await nova_engine.get_scene_metrics(scene_id)
    
    return SceneMetrics(
        scene_id=scene_id,
        uptime_seconds=metrics['uptime'],
        quantum_stability=metrics['quantum_stability'],
        user_satisfaction=metrics['user_satisfaction'],
        resource_usage=metrics['resource_usage']
    )

# ===== 用户API =====

@app.get(
    "/api/v1/users/me",
    response_model=UserInfo,
    tags=["Users"],
    summary="获取当前用户信息"
)
async def get_current_user_info(
    user_id: str = Depends(get_current_user),
    nova_engine: NovaEngine = Depends(get_nova_engine)
):
    """获取当前用户信息"""
    profile = await nova_engine.user_manager.get_profile(user_id)
    user_scenes = await nova_engine.get_user_scenes(user_id)
    active_scenes = len([s for s in user_scenes if s.status == 'active'])
    
    return UserInfo(
        user_id=user_id,
        tier=profile.tier,
        daily_budget=profile.daily_budget,
        used_budget=profile.used_budget,
        active_scenes=active_scenes
    )

# ===== WebSocket端点 =====

@app.websocket("/ws/scenes/{scene_id}")
async def scene_websocket(
    websocket: WebSocket,
    scene_id: str,
    nova_engine: NovaEngine = Depends(get_nova_engine)
):
    """
    场景实时通信WebSocket
    
    支持的消息类型：
    - interaction: 用户交互
    - adjust_quality: 调整质量
    - get_status: 获取状态
    """
    # 简单的认证（从查询参数获取token）
    token = websocket.query_params.get("token", "")
    user_id = token if token.startswith("user_") else "user_test_001"
    
    # 检查场景权限
    scene = await nova_engine.get_scene(scene_id)
    if not scene:
        await websocket.close(code=4404, reason="Scene not found")
        return
        
    if scene.user_id != user_id:
        await websocket.close(code=4403, reason="Access denied")
        return
        
    # 连接管理
    manager = app.state.connection_manager
    await manager.connect(websocket, scene_id, user_id)
    
    try:
        # 发送初始状态
        await websocket.send_json({
            "type": "connected",
            "scene_id": scene_id,
            "status": scene.status,
            "experience_level": scene.experience_level.value
        })
        
        # 消息循环
        while True:
            # 接收消息
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "interaction":
                # 处理交互
                action = data.get("action")
                parameters = data.get("parameters", {})
                
                # TODO: 实现交互处理
                # result = await nova_engine.process_interaction(
                #     scene_id, action, parameters
                # )
                
                # 广播更新
                await manager.broadcast_to_scene(scene_id, {
                    "type": "interaction_update",
                    "action": action,
                    "result": "processed"
                })
                
            elif message_type == "adjust_quality":
                # 调整质量等级
                new_level = data.get("level")
                
                # TODO: 实现质量调整
                # await nova_engine.adjust_scene_quality(scene_id, new_level)
                
                await websocket.send_json({
                    "type": "quality_adjusted",
                    "new_level": new_level,
                    "message": "Quality adjustment in progress"
                })
                
            elif message_type == "get_status":
                # 获取场景状态
                metrics = await nova_engine.get_scene_metrics(scene_id)
                
                await websocket.send_json({
                    "type": "status_update",
                    "metrics": metrics
                })
                
            elif message_type == "ping":
                # 心跳
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
                
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user={user_id}, scene={scene_id}")
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=4500, reason="Internal error")
        
    finally:
        manager.disconnect(websocket, scene_id)
        
        # 如果没有其他连接，暂停场景
        if not manager.has_connections(scene_id):
            try:
                await nova_engine.pause_scene(scene_id)
                logger.info(f"Scene {scene_id} paused (no active connections)")
            except Exception as e:
                logger.error(f"Error pausing scene: {e}")

# ===== 管理API =====

@app.post(
    "/api/v1/admin/broadcast",
    tags=["Admin"],
    include_in_schema=settings.DEBUG
)
async def broadcast_message(
    message: Dict[str, Any],
    scene_id: Optional[str] = None,
    user_id: str = Depends(get_current_user)
):
    """广播消息到场景（仅调试模式）"""
    if not settings.DEBUG:
        raise HTTPException(
            status_code=404,
            detail="Not found"
        )
        
    manager = app.state.connection_manager
    
    if scene_id:
        await manager.broadcast_to_scene(scene_id, message)
        return {"message": f"Broadcasted to scene {scene_id}"}
    else:
        # 广播到所有场景
        for sid in manager.active_connections.keys():
            await manager.broadcast_to_scene(sid, message)
        return {"message": "Broadcasted to all scenes"}

# ===== 工具函数 =====

async def notify_scene_created(scene_id: str, user_id: str):
    """通知场景创建（后台任务）"""
    # TODO: 实现通知逻辑
    # 例如：发送到消息队列、记录日志等
    logger.info(f"Scene {scene_id} created for user {user_id}")

# ===== 主函数 =====

if __name__ == "__main__":
    # 开发模式运行
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )