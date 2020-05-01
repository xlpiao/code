# Design Patterns with C++
## Reference
[Wikibooks: Design Patterns](https://en.wikibooks.org/wiki/C%2B%2B_Programming/Code/Design_Patterns)

[Sourcemaking: Design Patterns](https://sourcemaking.com/design_patterns)

## GoF: Gang of Four
The Gang of Four are the four authors of the book, "Design Patterns: Elements of Reusable Object-Oriented Software".

### Creational Patterns
Creational patterns are ones that create objects for you, rather than having you instantiate objects directly. This gives your program more flexibility in deciding which objects need to be created for a given case.

- [ ] 01.  **Abstract Factory** groups object factories that have a common theme.
- [x] 02.  **Builder** constructs complex objects by separating construction and representation.
- [ ] 03.  **Factory Method** creates objects without specifying the exact class to create.
- [x] 04.  **Prototype** creates objects by cloning an existing object.
- [x] 05.  **Singleton** restricts object creation for a class to only one instance.

### Structural Patterns
These concern class and object composition. They use inheritance to compose interfaces and define ways to compose objects to obtain new functionality.

- [ ] 01.  **Adapter** allows classes with incompatible interfaces to work together by wrapping its own interface around that of an already existing class.
- [ ] 02.  **Bridge** decouples an abstraction from its implementation so that the two can vary independently.
- [ ] 03.  **Composite** composes zero-or-more similar objects so that they can be manipulated as one object.
- [ ] 04.  **Decorator** dynamically adds/overrides behaviour in an existing method of an object.
- [ ] 05.  **Facade** provides a simplified interface to a large body of code.
- [ ] 06.  **Flyweight** reduces the cost of creating and manipulating a large number of similar objects.
- [ ] 07.  **Proxy** provides a placeholder for another object to control access, reduce cost, and reduce complexity.

### Behavioral Patterns
Most of these design patterns are specifically concerned with communication between objects.

- [ ] 01.  **Chain of Responsibility** delegates commands to a chain of processing objects.
- [ ] 02.  **Command** creates objects which encapsulate actions and parameters.
- [ ] 03.  **Interpreter** implements a specialized language.
- [ ] 04.  **Iterator** accesses the elements of an object sequentially without exposing its underlying representation.
- [ ] 05.  **Mediator** allows loose coupling between classes by being the only class that has detailed knowledge of their methods.
- [ ] 06.  **Memento** provides the ability to restore an object to its previous state (undo).
- [ ] 07.  **Observer** is a publish/subscribe pattern which allows a number of observer objects to see an event.
- [ ] 08.  **State** allows an object to alter its behavior when its internal state changes.
- [ ] 09.  **Strategy** allows one of a family of algorithms to be selected on-the-fly at runtime.
- [ ] 10.  **Template** defines the skeleton of an algorithm as an abstract class, allowing its subclasses to provide concrete behavior.
- [ ] 11.  **Visitor** separates an algorithm from an object structure by moving the hierarchy of methods into one object.
