import java.util.*;

public class Layer {

    private List<Neuron> neurons;

    public Layer(int nin, int nout){
        this.neurons = new ArrayList<>();
        for(int i = 0; i < nout; i++){
            neurons.add(new Neuron(nin));
        }
    }

    public List<Value> forward(List<Value> x){
        List<Value> outs = new ArrayList<>();
        for(int i = 0; i < neurons.size(); i++){
            Neuron n = neurons.get(i);
            outs.add(n.forward(x));
        }
        return outs;
    }

    public List<Neuron> getNeurons(){
        return this.neurons;
    }

    public List<Value> parameters(){
        List<Value> params = new ArrayList<>();
        for(int i = 0; i < neurons.size(); i++){
            params.addAll(neurons.get(i).parameters());
        }
        return params;
    }

    public void zeroGrad() {
        for (Neuron n : this.neurons) {
            n.zeroGrad();
        }
    }

}
