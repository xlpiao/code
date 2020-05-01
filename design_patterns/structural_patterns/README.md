# 1. Structural Patterns
Structural design patterns are design patterns that ease the design by identifying a simple way to realize relationships between entities.

- [ ] **2.1 Adapter**

- [ ] **2.2 Bridge**

- [ ] **2.3 Composite**

- [ ] **2.4 Decorator**

- [ ] **2.5 Facade**

- [ ] **2.6 Flyweight**

- [ ] **2.7 Proxy**


## 1.1 Adapter
Match interfaces of different classes

### Definition
Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces. Wrap an existing class with a new interface. Impedance match an old component to a new system

### Problem
We want to decide at run time what object is to be created based on some configuration or application parameter. When we write the code, we do not know what class should be instantiated.

### Solution
Define an interface for creating an object, but let subclasses decide which class to instantiate. Factory Method lets a class defer instantiation to subclasses.
