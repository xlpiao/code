class MinStack {
public:
  /** initialize your data structure here. */
  MinStack() {}

  void push(int val) {
    int min;

    if (st.empty()) {
      min = val;
    } else {
      min = std::min(st.top().second, val);
    }
    st.push({val, min});
  }

  void pop() { st.pop(); }

  int top() { return st.top().first; }

  int getMin() { return st.top().second; }

private:
  stack<pair<int, int>> st;
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(val);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
