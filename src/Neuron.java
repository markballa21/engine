import java.util.*;

public class Neuron {

    private List<Value> w;
    private Value bias;

    public Neuron(int nin){
        this.w = new ArrayList<>();
        Random rnd = new Random();
        for(int i = 0; i < nin; i++){
            w.add(new Value(rnd.nextDouble() * 2 - 1));
        }
        this.bias = new Value(0.0);
    }

    public Value forward(List<Value> inputs){
        // x1w1 + x2w2 +...+b
        Value sum = bias;
        for(int i = 0; i < inputs.size(); i++){
            Value v = w.get(i).mul(inputs.get(i));
            sum = sum.add(v);
        }
        return sum.tanh();
    }

    public List<Value> getW(){
        return this.w;
    }

    public List<Value> parameters(){
        List<Value> params = new ArrayList<>(w);
        params.add(bias);
        return params;
    }

    public void zeroGrad() {
        for (Value val : this.w) {
            val.setGrad(0.0);
        }
        this.bias.setGrad(0.0);
    }

}
