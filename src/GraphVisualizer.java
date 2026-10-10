
import java.util.*;

public class GraphVisualizer {

    public static class GraphData {
        public final Set<Value> nodes = new HashSet<>();
        public final Set<Map.Entry<Value, Value>> edges = new HashSet<>();
    }

    private static void trace(Value root, GraphData data) {
        if (!data.nodes.contains(root)) {
            data.nodes.add(root);
            for (Value child : root.getPrev()) {
                data.edges.add(new AbstractMap.SimpleEntry<>(child, root));
                trace(child, data);
            }
        }
    }

    public static String toDot(Value root) {
        GraphData data = new GraphData();
        trace(root, data);

        StringBuilder sb = new StringBuilder();
        sb.append("digraph G {\n");
        sb.append("  rankdir=\"LR\";\n");
        sb.append("  bgcolor=\"#FAFAFA\";\n");
        sb.append("  splines=polyline;\n");
        sb.append("  node [fontname=\"Helvetica,Arial,sans-serif\", fontsize=10, penwidth=1.2];\n");
        sb.append("  edge [color=\"#64748B\", penwidth=1.2, arrowsize=0.7];\n\n");

        // 1. Value Nodes
        for (Value n : data.nodes) {
            String uid = String.valueOf(System.identityHashCode(n));
            String name = n.getLabel().isEmpty() ? "" : n.getLabel() + " | ";
            String label = String.format("{ %sdata: %.4f | grad: %.4f }", name, n.getData(), n.getGrad());

            sb.append(String.format(
                    "  \"%s\" [shape=record, style=\"filled,rounded\", fillcolor=\"#FFFFFF\", color=\"#CBD5E1\", fontcolor=\"#1E293B\", label=\"%s\"];\n",
                    uid, label
            ));

            // Operation Nodes (circular badges)
            if (!n.getOp().isEmpty()) {
                String opId = uid + n.getOp();
                sb.append(String.format(
                        "  \"%s\" [label=\"%s\", shape=circle, style=\"filled\", fillcolor=\"#F1F5F9\", color=\"#94A3B8\", fontcolor=\"#0F172A\", fontsize=12, width=0.35, fixedsize=true];\n",
                        opId, n.getOp()
                ));
                sb.append(String.format("  \"%s\" -> \"%s\" [color=\"#94A3B8\"];\n", opId, uid));
            }
        }

        sb.append("\n");

        // 2. Directed Edges
        for (Map.Entry<Value, Value> edge : data.edges) {
            Value child = edge.getKey();
            Value parent = edge.getValue();
            String childId = String.valueOf(System.identityHashCode(child));

            if (!parent.getOp().isEmpty()) {
                String opId = String.valueOf(System.identityHashCode(parent)) + parent.getOp();
                sb.append(String.format("  \"%s\" -> \"%s\";\n", childId, opId));
            } else {
                String parentId = String.valueOf(System.identityHashCode(parent));
                sb.append(String.format("  \"%s\" -> \"%s\";\n", childId, parentId));
            }
        }

        sb.append("}\n");
        return sb.toString();
    }
}