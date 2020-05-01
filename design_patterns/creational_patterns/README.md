# 1. Creational Patterns
These design patterns are all about class instantiation. This pattern can be further divided into class-creation patterns and object-creational patterns. While class-creation patterns use inheritance effectively in the instantiation process, object-creation patterns use delegation effectively to get the job done.

- [x] **1.1 Abstract Factory Pattern**

- [x] **1.2 Builder Pattern**

- [ ] **1.3 Factory Method Pattern**

- [ ] **1.4 Object Pool Pattern**

- [x] **1.5 Prototype Pattern**

- [x] **1.6 Singleton Pattern**


## 1.1 Abstract Factory Pattern
Creates an instance of several families of classes

### Definition
A utility class that creates an instance of several families of classes. It can also return a factory for a certain group. The Factory Design Pattern is useful in a situation that requires the creation of many different types of objects, all derived from a common base type. The Factory Method defines a method for creating the objects, which subclasses can then override to specify the derived type that will be created. Thus, at run time, the Factory Method can be passed a description of a desired object (e.g., a string read from user input) and return a base class pointer to a new instance of that object. The pattern works best when a well-designed interface is used for the base class, so there is no need to cast the returned object.

### Problem
We want to decide at run time what object is to be created based on some configuration or application parameter. When we write the code, we do not know what class should be instantiated.

### Solution
Define an interface for creating an object, but let subclasses decide which class to instantiate. Factory Method lets a class defer instantiation to subclasses.


## 1.2 Builder Pattern
Separates object construction from its representation

### Definition
The Builder Creational Pattern is used to separate the construction of a complex object from its representation so that the same construction process can create different objects representations.

### Problem
We want to construct a complex object, however we do not want to have a complex constructor member or one that would need many arguments.

### Solution
Define an intermediate object whose member functions define the desired object part by part before the object is available to the client. Builder Pattern lets us defer the construction of the object until all the options for creation have been specified.

## 1.3 Factory Method Pattern
Creates an instance of several derived classes

## 1.4 Object Pool Pattern
Avoid expensive acquisition and release of resources by recycling objects that are no longer in use

## 1.5 Prototype Pattern
A fully initialized instance to be copied or cloned

### Definition
A prototype pattern is used in software development when the type of objects to create is determined by a prototypical instance, which is cloned to produce new objects.

## 1.6 Singleton Pattern
A class of which only a single instance can exist

### Definition
The Singleton pattern ensures that a class has only one instance and provides a global point of access to that instance. It is named after the singleton set, which is defined to be a set containing one element. This is useful when exactly one object is needed to coordinate actions across the system.
