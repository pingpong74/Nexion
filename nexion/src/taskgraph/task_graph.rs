use crate::{Barrier, BufferID, Device, ImageID, ImageViewID, Swapchain, taskgraph::definations::*};

/// Pre compliation task graph.
/// It can be mutated and all resources required must be specified on this stage
/// After compiling it turns into an executable task graph. Resources can only be modified and not added
pub struct TaskGraph {
    device: Device,
    swapchain: Option<Swapchain>,
    tasks: Vec<Task>,
    // store actual presistent resources
    images: Vec<ImageID>,
    buffers: Vec<BufferID>,
    image_views: Vec<ImageViewID>,
}

impl TaskGraph {
    pub fn new(task_graph_desc: TaskGraphDescription) -> TaskGraph {
        return TaskGraph {
            device: task_graph_desc.device,
            swapchain: task_graph_desc.swapchain,
            tasks: vec![],
            images: vec![],
            buffers: vec![],
            image_views: vec![],
        };
    }

    /// Adds a new Image to the task graph
    pub fn use_image(&mut self, image_id: ImageID) -> TaskImageId {
        self.images.push(image_id);

        return TaskImageId(self.images.len() - 1);
    }

    /// Adds a new Buffer to the task graph
    pub fn use_buffer(&mut self, buffer_id: BufferID) -> TaskBufferId {
        self.buffers.push(buffer_id);

        return TaskBufferId(self.buffers.len() - 1);
    }

    /// Adds new image view slot
    pub fn use_image_view(&mut self, image_view_id: ImageViewID) -> TaskImageViewId {
        self.image_views.push(image_view_id);

        return TaskImageViewId(self.image_views.len() - 1);
    }

    pub fn add_task(&mut self, task: Task) {
        self.tasks.push(task);
    }

    pub fn preset(&self) {
        !unimplemented!()
    }

    pub fn submit(&self) {
        !unimplemented!()
    }

    pub fn compile(self) {
        let adj_list = self.create_adjacency_list();
        let batches = TaskGraph::toplogical_sort(&adj_list);
    }
}

impl TaskGraph {
    fn check_dependencies(a: &Task, b: &Task) -> bool {
        for res_a in &a.resources {
            for res_b in &b.resources {
                if TaskResource::same_resource(res_a, res_b) {
                    // Extract access types
                    let access_a = res_a.get_access();
                    let access_b = res_b.get_access();

                    // Dependency if not both are reads
                    if !(matches!(access_a, TaskAccess::Read) && matches!(access_b, TaskAccess::Read)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    fn create_adjacency_list(&self) -> Vec<Vec<usize>> {
        let mut edges = vec![vec![]; self.tasks.len()];

        for (i, pass_i) in self.tasks.iter().enumerate() {
            for j in 0..i {
                let pass_j = &self.tasks[j];

                if TaskGraph::check_dependencies(pass_j, pass_i) {
                    edges[i].push(j);
                }
            }
        }

        TaskGraph::transitive_reduction(&mut edges);

        return edges;
    }

    fn transitive_reduction(adj: &mut Vec<Vec<usize>>) {
        let n = adj.len();

        for u in 0..n {
            let neighbors = adj[u].clone();
            for &v in &neighbors {
                adj[u].retain(|&x| x != v);

                let is_reachable = {
                    let mut visited = vec![false; adj.len()];
                    let mut stack = vec![u];

                    let mut b = false;

                    while let Some(node) = stack.pop() {
                        if node == v {
                            b = true;
                            break;
                        }
                        if visited[node] {
                            continue;
                        }
                        visited[node] = true;
                        for &next in &adj[node] {
                            stack.push(next);
                        }
                    }
                    b
                };

                if !is_reachable {
                    adj[u].push(v);
                }
            }
        }
    }

    fn toplogical_sort(adj_list: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        let mut indegrees = vec![0; adj_list.len()];
        for u in 0..adj_list.len() {
            for &v in &adj_list[u] {
                indegrees[v] += 1;
            }
        }

        let mut q = std::collections::VecDeque::new();
        let mut batches = Vec::new();

        for i in 0..adj_list.len() {
            if indegrees[i] == 0 {
                println!("{}", i);
                q.push_back(i);
            }
        }

        while !q.is_empty() {
            let mut batch = Vec::new();
            let size = q.len();

            for _ in 0..size {
                let u = q.pop_front().unwrap();
                batch.push(u);

                for &v in &adj_list[u] {
                    indegrees[v] -= 1;
                    if indegrees[v] == 0 {
                        q.push_back(v);
                    }
                }
            }

            batches.push(batch);
        }

        batches.reverse();

        return batches;
    }

    // Maybe try per resource? lets see that makes more sense.
    fn generate_barriers(&self, batches: &Vec<Vec<usize>>, adj_list: &Vec<Vec<usize>>) -> Vec<Vec<Barrier>> {
        for i in 0..(batches.len() - 1) {
            for pass_index in &batches[i] {}
        }

        unimplemented!()
    }
}

pub struct ExecutableTaskGraph {
    device: Device,
    swapchain: Option<Swapchain>,
    // execution info, recording functions and barriers
    barriers: Vec<Barrier>,
    tasks: Vec<Box<dyn Fn(&TaskGraphInterface) + 'static>>,
    // store actual presistent resources
    images: Vec<ImageID>,
    buffers: Vec<BufferID>,
    image_views: Vec<ImageViewID>,
}

impl ExecutableTaskGraph {
    /// Updates a prexisting image slot
    pub fn update_image(&mut self, task_image_id: TaskImageId, image_id: ImageID) {
        self.images[task_image_id.0] = image_id;
    }

    /// Updates a prexisting buffer slot
    pub fn update_buffer(&mut self, task_buffer_id: TaskBufferId, buffer_id: BufferID) {
        self.buffers[task_buffer_id.0] = buffer_id;
    }

    /// Updates a prexisting image view slot
    pub fn update_image_view(&mut self, task_image_view_id: TaskImageViewId, image_view_id: ImageViewID) {
        self.image_views[task_image_view_id.0] = image_view_id;
    }

    pub fn execute(&self) {}
}
