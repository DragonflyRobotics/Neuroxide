use neuroxide::ops::op_generic::AddOp;
use neuroxide::types::tensor::Tensor;
use neuroxide::types::device::Device;
use neuroxide::ops::op_generic::Operation;
use petgraph::dot::Dot;
use petgraph::prelude::GraphMap;
use petgraph::graph::NodeIndex;

fn makeagraph() -> GraphMap<NodeIndex, (), petgraph::Directed> {
    let mut graph = GraphMap::new(); 
    let a = graph.add_node(NodeIndex::new(1));
    let b = graph.add_node(NodeIndex::new(2));
    graph.add_edge(a, b, ());
    println!("{:?}", Dot::new(&graph));
    graph
}

fn makebgraph() {
    let mut graph = petgraph::Graph::<&str, &str>::new();
    let a = graph.add_node("a");
    let b = graph.add_node("b");
    let c = graph.add_node("c");
    let d = graph.add_node("d");
    graph.add_edge(a, b, "ab");
    graph.add_edge(b, c, "bc");
    graph.add_edge(c, d, "cd");
    graph.add_edge(d, a, "da");
    println!("{:?}", Dot::new(&graph));
}

fn main() {
    let mut a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], Device::CPU, true);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3], Device::CPU, true);
    a.op_chain = makeagraph();
    
    let add_op = AddOp;
    let result = add_op.forward(&vec![a, b]);
    println!("{:?}", result);
}
