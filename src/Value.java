import java.util.*;

public class Value {

    private double data;
    private double grad;
    private final List<Value> prev; // children nodes
    private final String op;
    private String label;
    private Runnable backwardStep;

    public Value(double data, List<Value> children, String op, String label){
        this.data = data;
        this.grad = 0.0;           // always initialize 0
        this.prev = children != null ? children : Collections.emptyList();
        this.op = op;
        this.label = (label != null) ? label : "";
        this.backwardStep = () -> {};
    }

    public Value(double data, List<Value> children, String op) {
        this(data, children, op, "");
    }


    public Value(double data, String label) {
        this(data, Collections.emptyList(), "", label);
    }


    public Value(double data) {
        this(data, Collections.emptyList(), "", "");
    }

    // Getters & Setters
    public double getData() { return this.data; }
    public void setData(double data) { this.data = data; }
    public double getGrad() { return this.grad; }
    public void setGrad(double grad) { this.grad = grad; }
    public void addGrad(double val) { this.grad += val; }

    public List<Value> getPrev() { return prev; }
    public String getOp() { return op; }
    public String getLabel() { return label; }
    public void setLabel(String label) { this.label = label; }

    public Value add(Value other){
        List<Value> children = new ArrayList<>(Arrays.asList(this, other));
        Value out = new Value(this.data + other.getData(), children, "+");
        out.backwardStep = () -> {
            this.grad += 1.0 * out.grad;
            other.grad += 1.0 * out.grad;
        };
        return out;
    }

    public Value pow(double other) {
        Value out = new Value(Math.pow(this.data, other), List.of(this), "**" + other);
        out.backwardStep = () -> {
            // d/dx [x^k] = k * x^(k-1)
            this.grad += out.grad * (other * Math.pow(this.data, other - 1.0));
        };

        return out;
    }

    public Value neg() {
        return this.mul(new Value(-1.0)); // Uses existing mul backward
    }

    public Value sub(Value other) {
        return this.add(other.neg()); // Uses existing add backward
    }

    public Value mul(Value other){
        List<Value> children = new ArrayList<>(Arrays.asList(this, other));
        Value out = new Value(this.data * other.getData(), children, "*");
        out.backwardStep = () -> {
            this.grad += other.data * out.grad;
            other.grad += this.data * out.grad;
        };
        return out;
    }

    public Value tanh(){
        List<Value> children = new ArrayList<>(List.of(this));
        double t = Math.tanh(this.data);
        Value out = new Value(t, children, "tanh");
        out.backwardStep = () -> {
            this.grad += (1.0 - t * t) * out.grad; // d/dx tanh(x) = 1 - tanh(x)^2
        };
        return out;
    }

    public void backward() {
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();

        buildTopo(this, visited, topo);

        // Seed gradient of the output node
        this.grad = 1.0;

        // Traverse in reverse topological order
        Collections.reverse(topo);
        for (Value v : topo) {
            v.backwardStep.run();
        }
    }

    private void buildTopo(Value v, Set<Value> visited, List<Value> topo) {
        if (!visited.contains(v)) {
            visited.add(v);
            for (Value child : v.prev) {
                buildTopo(child, visited, topo);
            }
            topo.add(v);
        }
    }

    @Override
    public String toString() {
        return "Value(data=" + this.data + ")";
    }
}
