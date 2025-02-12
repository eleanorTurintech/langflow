from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING

from loguru import logger

from langflow.utils.concurrency import KeyedMemoryLockManager

if TYPE_CHECKING:
    from langflow.services.base import Service
    from langflow.services.factory import ServiceFactory
    from langflow.services.schema import ServiceType


class NoFactoryRegisteredError(Exception):
    pass


class ServiceManager:
    """Manages the creation of different services."""

from typing import Dict, Type
from types import SimpleNamespace
from threading import Lock

class Service:
    pass

class ServiceFactory:
    pass

class KeyedMemoryLockManager:
    pass

def __init__(self) -> None:
    # Pre-allocate dictionaries with explicit typing for better memory management
    self.services: Dict[str, Service] = {}
    self.factories: Dict[str, ServiceFactory] = {}
    # Initialize service factories
    self.register_factories()
    # Initialize the keyed lock manager
    self.keyed_lock = KeyedMemoryLockManager()

from typing import Dict, Set, Any, Generator
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class ServiceRegistrar:
    def __init__(self):
        """Initialize the registrar with cache and dependency tracking."""
        self._dependency_cache: Dict[str, Any] = {}
        self._resolved_factories: Set[str] = set()
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
    def register_factories(self) -> None:
        """Registers service factories.

        Iterates through each factory obtained from `get_factories()`,
        attempts to register it, and logs any exceptions encountered during
        the registration process.  This implementation assumes `get_factories()`
        is a generator to minimize memory usage by processing factories one
        by one.

        """
        processing_stack: Set[str] = set()  # Track factories being processed
        
        for factory in self.get_factories():
            try:
                factory_id = self._get_factory_id(factory)
                if factory_id in self._resolved_factories:
                    continue
                    
                # Check for circular dependencies
                if factory_id in processing_stack:
                    raise ValueError(f"Circular dependency detected for {factory_id}")
                    
                processing_stack.add(factory_id)
                
                # Resolve dependencies first
                self._resolve_dependencies(factory)
                
                # Attempt to register the current factory
                self.register_factory(factory)
                
                # Cache the resolved factory
                self._resolved_factories.add(factory_id)
                processing_stack.remove(factory_id)
                
                # Remove reference to the factory after registration
                del factory
                
            except Exception:  # noqa: BLE001
                # Log any exceptions that occur during factory initialization.
                logger.exception(f"Error initializing {factory}")

    def get_factories(self) -> Generator[str, None, None]:  # Example generator implementation
        """
        Yields service factories one at a time.

        This generator simulates retrieving factories from a source.
        Replace this with the actual implementation to fetch factories as needed.
        It minimizes memory usage compared to returning a list of all factories at once.

        Yields:
            Any: A service factory.

        """
        # Replace this with the actual factory retrieval logic
        for i in range(5):  # Simulate having 5 factories.
            yield f"Factory_{i}"

    def register_factory(self, factory) -> None:
        """
        Registers a single service factory.

        This method performs the registration logic for a given factory.
        Replace this with the actual implementation of factory registration.

        Args:
            factory (Any): The service factory to register.

        """
        # Replace this with the actual factory registration logic
        print(f"Registered: {factory}")  # Simulate registration
        
    def _get_factory_id(self, factory: Any) -> str:
        """Generate a unique identifier for the factory."""
        return str(factory)
        
    def _resolve_dependencies(self, factory: Any) -> None:
        """
        Resolve and cache dependencies for a factory.
        
        Args:
            factory (Any): The factory whose dependencies need to be resolved.
        """
        factory_id = self._get_factory_id(factory)
        
        # Return if dependencies are already cached
        if factory_id in self._dependency_cache:
            return
            
        # Get factory dependencies (implement this based on your needs)
        dependencies = self._get_factory_dependencies(factory)
        
        # Cache the resolved dependencies
        self._dependency_cache[factory_id] = dependencies
        
    def _get_factory_dependencies(self, factory: Any) -> Set[str]:
        """
        Get the dependencies for a factory.
        
        Args:
            factory (Any): The factory to get dependencies for.
            
        Returns:
            Set[str]: Set of dependency identifiers.
        """
        # Replace this with actual dependency resolution logic
        return set()  # Return empty set for this example

    def register_factory(
        self,
        service_factory: ServiceFactory,
    ) -> None:
        """Registers a new factory with dependencies."""
        service_name = service_factory.service_class.name
        self.factories[service_name] = service_factory

    def get(self, service_name: ServiceType, default: ServiceFactory | None = None) -> Service:
        """Get (or create) a service by its name."""
        with self.keyed_lock.lock(service_name):
            if service_name not in self.services:
                self._create_service(service_name, default)

        return self.services[service_name]

    def _create_service(self, service_name: ServiceType, default: ServiceFactory | None = None) -> None:
        """Create a new service given its name, handling dependencies."""
        logger.debug(f"Create service {service_name}")
        self._validate_service_creation(service_name, default)

        # Create dependencies first
        factory = self.factories.get(service_name)
        if factory is None and default is not None:
            self.register_factory(default)
            factory = default
        if factory is None:
            msg = f"No factory registered for {service_name}"
            raise NoFactoryRegisteredError(msg)
        for dependency in factory.dependencies:
            if dependency not in self.services:
                self._create_service(dependency)

        # Collect the dependent services
        dependent_services = {dep.value: self.services[dep] for dep in factory.dependencies}

        # Create the actual service
        self.services[service_name] = self.factories[service_name].create(**dependent_services)
        self.services[service_name].set_ready()

    def _validate_service_creation(self, service_name: ServiceType, default: ServiceFactory | None = None) -> None:
        """Validate whether the service can be created."""
        if service_name not in self.factories and default is None:
            msg = f"No factory registered for the service class '{service_name.name}'"
            raise NoFactoryRegisteredError(msg)

    def update(self, service_name: ServiceType) -> None:
        """Update a service by its name."""
        if service_name in self.services:
            logger.debug(f"Update service {service_name}")
            self.services.pop(service_name, None)
            self.get(service_name)

    async def teardown(self) -> None:
        """Teardown all the services."""
        for service in self.services.values():
            if service is None:
                continue
            logger.debug(f"Teardown service {service.name}")
            try:
                await service.teardown()
            except Exception as exc:  # noqa: BLE001
                logger.exception(exc)
        self.services = {}
        self.factories = {}

    @staticmethod
import asyncio
import importlib
import inspect
from typing import Any, Dict, Generic, List, Type, TypeVar

from langflow.services.factory import ServiceFactory
from langflow.services.schema import ServiceType
from loguru import logger
import concurrent.futures

T = TypeVar("T", bound=ServiceFactory)


class ServicePool(Generic[T]):
    def __init__(self, factory: Type[T], pool_size: int = 5):
        self.factory: Type[T] = factory
        self.pool: List[T] = []
        self.pool_size = pool_size
        self._expand_pool(pool_size)

    async def _aexpand_pool(self, count: int):
        async def create_service():
            return self.factory()
        tasks = [create_service() for _ in range(count)]
        self.pool.extend(await asyncio.gather(*tasks))


    def _expand_pool(self, count: int):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._aexpand_pool(count))



    def acquire(self) -> T:
        if not self.pool:
            self._expand_pool(1)  # Expand on demand if pool is empty
        return self.pool.pop()

    def release(self, instance: T):
        if len(self.pool) < self.pool_size:
            self.pool.append(instance)
        # If the pool is full, let the instance be garbage collected

    def teardown(self):
        """Shuts down all services in the pool concurrently."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(service.teardown) for service in self.pool]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Retrieve any potential exceptions
                except Exception as e:
                    logger.error(f"Error during service teardown: {e}")
        self.pool.clear()




# Define a pool registry to hold instances of ServicePool
_service_pools: Dict[Type[ServiceFactory], ServicePool] = {}


def get_service(factory_class: Type[T]) -> T:
    """Retrieves a service instance from the pool, or creates a new one if needed."""
    pool = _service_pools.get(factory_class)
    if pool is None:
        pool = ServicePool(factory_class)
        _service_pools[factory_class] = pool
    return pool.acquire()


def release_service(instance: ServiceFactory):
    """Releases a service instance back to its respective pool."""
    for factory_class, pool in _service_pools.items():
        if isinstance(instance, factory_class):
            pool.release(instance)
            return

    logger.warning(f"Attempting to release an unknown service type: {type(instance)}")



async def _aget_factories():
    from langflow.services.factory import ServiceFactory
    from langflow.services.schema import ServiceType

    service_names = [ServiceType(service_type).value.replace("_service", "") for service_type in ServiceType]
    base_module = "langflow.services"
    factories = []

    async def process_service(name):

        try:
            module_name = f"{base_module}.{name}.factory"
            module = importlib.import_module(module_name)

            # Find all classes in the module that are subclasses of ServiceFactory
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, ServiceFactory) and obj is not ServiceFactory:
                    # Changed to use the pool instead of creating instances directly.
                    return obj

        except Exception as exc:
            logger.exception(exc)
            msg = f"Could not initialize services. Please check your settings. Error in {name}."
            raise RuntimeError(msg) from exc
        return None
    results = await asyncio.gather(*[process_service(name) for name in service_names])

    # Filter out any None results that might have occurred due to exceptions
    factories.extend([factory for factory in results if factory is not None])

    return factories

def get_factories():
    """Retrieves all service factories."""
    async def get_factories_async():
        return await _aget_factories()
    return asyncio.run(get_factories_async())





def teardown_service_pools():
    """Tears down all service pools, shutting down their services concurrently."""
    for pool in _service_pools.values():
        pool.teardown()
    _service_pools.clear()


service_manager = ServiceManager()


def initialize_settings_service() -> None:
    """Initialize the settings manager."""
    from langflow.services.settings import factory as settings_factory

    service_manager.register_factory(settings_factory.SettingsServiceFactory())


def initialize_session_service() -> None:
    """Initialize the session manager."""
    from langflow.services.cache import factory as cache_factory
    from langflow.services.session import factory as session_service_factory

    initialize_settings_service()

    service_manager.register_factory(cache_factory.CacheServiceFactory())

    service_manager.register_factory(session_service_factory.SessionServiceFactory())