import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import deque

# ---------------------------------------------------------
# 1. EllipticalDBSCAN 클래스 (제안 알고리즘)
# ---------------------------------------------------------
class EllipticalDBSCAN:
    def __init__(self, L, min_pts, angle_threshold_degrees, min_branch_size):
        self.L = L
        self.min_pts = min_pts
        self.angle_threshold = np.radians(angle_threshold_degrees)
        self.min_branch_size = min_branch_size
        self.labels = {}
        self.visited = set()
        self.cluster_map = {}
        self.cluster_id_counter = 0

    def get_angle(self, p1, p2):
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    def get_angle_diff(self, theta1, theta2):
        diff = abs(theta1 - theta2)
        if diff > np.pi:
            diff = 2 * np.pi - diff
        return diff

    def check_density(self, p1, p2, data):
        count = 0
        for x in data:
            if np.linalg.norm(x - p1) + np.linalg.norm(x - p2) <= self.L:
                count += 1
        return count >= self.min_pts

    def fit(self, data):
        self.labels = {i: -1 for i in range(len(data))}
        self.visited = set()
        self.cluster_id_counter = 0

        for i in range(len(data)):
            if i in self.visited: continue
            
            initial_neighbors = []
            for j in range(len(data)):
                if i == j: continue
                if np.linalg.norm(data[i] - data[j]) <= self.L:
                    initial_neighbors.append(j)
            
            seed_found = False
            for j in initial_neighbors:
                if j in self.visited: continue
                if self.check_density(data[i], data[j], data):
                    self.cluster_id_counter += 1
                    current_cluster = self.cluster_id_counter
                    self.labels[i] = current_cluster
                    self.labels[j] = current_cluster
                    self.visited.add(i)
                    self.visited.add(j)
                    self._expand_cluster(data, [(i, j)], current_cluster)
                    seed_found = True
                    break
            
            if not seed_found:
                self.labels[i] = -1

        self._post_process_merge()
        return np.array([self.labels[i] for i in range(len(data))])

    def _expand_cluster(self, data, queue, current_cluster_id):
        q = deque(queue)
        while q:
            prev_idx, curr_idx = q.popleft()
            p_prev = data[prev_idx]
            p_curr = data[curr_idx]
            base_angle = self.get_angle(p_prev, p_curr)

            candidates = []
            for k in range(len(data)):
                if k == curr_idx or k == prev_idx: continue
                if k in self.visited: continue
                if np.linalg.norm(data[k] - p_curr) <= self.L:
                    candidates.append(k)

            for next_idx in candidates:
                p_next = data[next_idx]
                if not self.check_density(p_curr, p_next, data): continue

                new_angle = self.get_angle(p_curr, p_next)
                angle_diff = self.get_angle_diff(base_angle, new_angle)

                if angle_diff <= self.angle_threshold:
                    self.labels[next_idx] = current_cluster_id
                    self.visited.add(next_idx)
                    q.append((curr_idx, next_idx))
                else:
                    self.cluster_id_counter += 1
                    new_cluster_id = self.cluster_id_counter
                    self.labels[next_idx] = new_cluster_id
                    self.visited.add(next_idx)
                    self.cluster_map[new_cluster_id] = current_cluster_id
                    self._expand_cluster(data, [(curr_idx, next_idx)], new_cluster_id)

    def _post_process_merge(self):
        counts = {}
        for label in self.labels.values():
            if label == -1: continue
            counts[label] = counts.get(label, 0) + 1
            
        merged_something = True
        while merged_something:
            merged_something = False
            for child_id, parent_id in list(self.cluster_map.items()):
                if child_id in counts and counts[child_id] < self.min_branch_size:
                    if parent_id not in counts: continue
                    
                    for idx, label in self.labels.items():
                        if label == child_id: self.labels[idx] = parent_id
                    
                    counts[parent_id] += counts[child_id]
                    del counts[child_id]
                    del self.cluster_map[child_id]
                    
                    for sub_child, sub_parent in list(self.cluster_map.items()):
                        if sub_parent == child_id: self.cluster_map[sub_child] = parent_id
                    
                    merged_something = True

# ---------------------------------------------------------
# 2. 데이터 생성 (Y자 분기 + 노이즈)
# ---------------------------------------------------------
np.random.seed(42)
x1 = np.linspace(0, 10, 50); y1 = np.zeros_like(x1)
road1 = np.column_stack([x1, y1])

x2 = np.linspace(10, 15, 30); y2 = (x2 - 10) * np.tan(np.radians(30))
road2 = np.column_stack([x2, y2])

x3 = np.linspace(10, 15, 30); y3 = (x3 - 10) * np.tan(np.radians(-45))
road3 = np.column_stack([x3, y3])

noise = np.random.uniform(low=[0, -5], high=[15, 5], size=(30, 2))
X = np.vstack([road1, road2, road3, noise])
X += np.random.normal(0, 0.1, X.shape)

# ---------------------------------------------------------
# 3. 모델 비교 및 시각화
# ---------------------------------------------------------

# A. 제안 알고리즘 실행
# (L=2, min_pts=5, 35도 이내 진행, 10개 이하는 환원)
ed_model = EllipticalDBSCAN(L=1.5, min_pts=5, angle_threshold_degrees=30, min_branch_size=7)
ed_labels = ed_model.fit(X)

# B. 표준 DBSCAN 실행
# eps는 L의 절반 정도인 1.0으로 설정 (타원 장축 L=2와 비슷한 스케일)
std_model = DBSCAN(eps=1.0, min_samples=5)
std_labels = std_model.fit_predict(X)

# 시각화 함수
def plot_result(ax, data, labels, title):
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'; label_name = "Noise"; marker = 'x'; size = 20
        else:
            label_name = f"Cluster {k}"; marker = 'o'; size = 60
        
        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=[col], s=size, marker=marker, edgecolor='k', label=label_name)
    
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)

# 플롯 그리기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

plot_result(ax1, X, ed_labels, "Proposed: Slope-Sensitive Elliptical DBSCAN")
plot_result(ax2, X, std_labels, "Standard: DBSCAN (eps=1.0)")

plt.tight_layout()
plt.show()