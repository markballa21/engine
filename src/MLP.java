import java.util.*;

public class MLP {

    private List<Layer> layers;

    public MLP(int nin, List<Integer> nouts){
        List<Integer> sz = new ArrayList<>();
        sz.add(nin);
        sz.addAll(nouts);
        layers = new ArrayList<>();
        for(int i = 0; i < sz.size() - 1; i++){
            layers.add(new Layer(sz.get(i), sz.get(i+1)));
        }
    }

    public List<Value> forward(List<Value> x){
        for(Layer l: layers){
            x = l.forward(x);
        }
        return x;
    }

    public List<Layer> getLayers(){
        return this.layers;
    }

    public List<Value> parameters(){
        List<Value> params = new ArrayList<>();
        for(int i = 0; i < layers.size(); i++){
            params.addAll(layers.get(i).parameters());
        }
        return params;
    }

    public void zeroGrad(){
        for(Layer l: layers){
            l.zeroGrad();
        }

    }
}
