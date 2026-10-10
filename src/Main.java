import java.util.*;

public class Main {

    public static void main(String[] args){
        test8();
    }

    public static void test8() {
        // 1. Define network architecture [3 inputs -> 4 -> 4 -> 1 output]
        List<Integer> sizes = List.of(4, 4, 1);
        MLP m = new MLP(3, sizes);

        // 2. Initialize inputs xs: 4 samples, each with 3 features
        List<List<Value>> xs = List.of(
                List.of(new Value(2.0), new Value(3.0), new Value(-1.0)),
                List.of(new Value(3.0), new Value(-1.0), new Value(0.5)),
                List.of(new Value(0.5), new Value(1.0), new Value(1.0)),
                List.of(new Value(1.0), new Value(1.0), new Value(-1.0))
        );

        // 3. Desired target values ys
        List<Value> ys = List.of(
                new Value(1.0),
                new Value(-1.0),
                new Value(-1.0),
                new Value(1.0)
        );

        for (int j = 0; j < 30; j++) {

            // 1. Forward pass
            List<Value> ypred = new ArrayList<>();
            for (List<Value> x : xs) {
                ypred.add(m.forward(x).get(0));
            }

            // 2. Compute Loss
            Value loss = new Value(0.0);
            for (int i = 0; i < ys.size(); i++) {
                Value diff = ypred.get(i).sub(ys.get(i));
                loss = loss.add(diff.pow(2.0));
            }

            // Zero gradients from previous iteration
            m.zeroGrad();

            loss.backward();

            for (Value p : m.parameters()) {
                p.setData(p.getData() - 0.05 * p.getGrad());
            }

            System.out.printf("Step %d | Loss: %.4f%n", j, loss.getData());
        }

        //System.out.println(m.getLayers().get(0).getNeurons().get(0).getW().get(0).getGrad());
        //System.out.println(m.getLayers().get(0).getNeurons().get(0).getW().get(0).getData());
        //System.out.println(GraphVisualizer.toDot(loss));
        //System.out.println(m.getLayers().get(0).getNeurons().get(0).getW().get(0).getData());

    }

    public static void test7(){
        Value a = new Value(2, "a");
        Value b = new Value(3, "b");
        Value c = new Value(-1, "c");
        List<Value> l = new ArrayList<>(); l.add(a); l.add(b); l.add(c);
        List<Integer> sizes = new ArrayList<>(); sizes.add(4); sizes.add(4); sizes.add(1);
        MLP m = new MLP(3, sizes);
        System.out.println(m.forward(l));
        System.out.println(GraphVisualizer.toDot(m.forward(l).getFirst()));
    }

    public static void test6(){
        Value a = new Value(2, "a");
        Value b = new Value(3, "b");
        List<Value> l = new ArrayList<>(); l.add(a); l.add(b);
        Layer L = new Layer(2, 3);
        System.out.println(L.forward(l));
    }

    public static void test5(){
        Value a = new Value(2, "a");
        Value b = new Value(3, "b");
        List<Value> l = new ArrayList<>(); l.add(a); l.add(b);
        Neuron n1 = new Neuron(2);
        System.out.println(n1.forward(l));
    }

    public static void test4(){
        Value a = new Value(1, "a");
        Value b = a.add(a); b.setLabel("b");
        b.backward();
        System.out.println(a.getGrad()); // should be 2
    }

    public static void test3(){
        Value x1 = new Value(2, "x1");
        Value x2 = new Value(0, "x2");
        Value w1 = new Value(-3, "w1");
        Value w2 = new Value(1, "w2");
        Value b = new Value(6.88, "b");
        Value x1w1 = x1.mul(w1); x1w1.setLabel("x1*w1");
        Value x2w2 = x2.mul(w2); x2w2.setLabel("x2*w2");
        Value x1w1x2w2 = x1w1.add(x2w2); x1w1x2w2.setLabel("x1*w1 + x2*w2");
        Value n = x1w1x2w2.add(b); n.setLabel("n");
        Value o = n.tanh(); o.setLabel("o"); o.setGrad(1);
        o.backward();
        System.out.println(GraphVisualizer.toDot(o));
    }

    public static void test2(){
        Value x1 = new Value(2, "x1");
        Value x2 = new Value(0, "x2");
        Value w1 = new Value(-3, "w1");
        Value w2 = new Value(1, "w2");
        Value b = new Value(6.88, "b");
        Value x1w1 = x1.mul(w1); x1w1.setLabel("x1*w1");
        Value x2w2 = x2.mul(w2); x2w2.setLabel("x2*w2");
        Value x1w1x2w2 = x1w1.add(x2w2); x1w1x2w2.setLabel("x1*w1 + x2*w2");
        Value n = x1w1x2w2.add(b); n.setLabel("n");
        Value o = n.tanh(); o.setLabel("o"); o.setGrad(1);
        n.setGrad(0.5); // do/dn = 1- tanh^2 = 1 * 0.5
        b.setGrad(0.5); x1w1x2w2.setGrad(0.5); // dn/db = 1 * 0.5 dn/dx1w1x2w2 = 1 * 0.5
        x1w1.setGrad(0.5); x2w2.setGrad(0.5);  //dx1w1x2w2/x1w1 = 1 * 0.5  dx1w1x2w2/x2w2 = 1 * 0.5
        x1.setGrad(-1.5); w1.setGrad(1); //dx1w1/dx1 = w1 * 0.5 = -1.5   dx1w1/dw1 = x1 * 0.5 = 1
        x2.setGrad(0.5); w2.setGrad(0); //dx2w2/dx2 = w2 * 0.5 = 0.5   dx2w2/dw2 = x2 * 0.5 = 0
        System.out.println(GraphVisualizer.toDot(o));
    }

    public static void test1(){
        Value a = new Value(2, "a");
        Value b = new Value(-3, "b");
        Value c = new Value(10, "c");
        Value e = a.mul(b); e.setLabel("e");
        Value d = e.add(c); d.setLabel("d");
        Value f = new Value(-2, "f");
        Value L = d.mul(f); L.setLabel("L");
        L.setGrad(1); // L = d * f
        f.setGrad(4); d.setGrad(-2); // dL/dd = f, dL/df = d
        e.setGrad(-2); c.setGrad(-2); // dL/dc = dL/dd * dd/dc = f * 1 = f
        a.setGrad(-2 * 2); b.setGrad(-2 * -3); // dL/da = dL/dd * dd/dc * de/da = f * 1 * b = f*b
        System.out.println(GraphVisualizer.toDot(L));
    }
}
