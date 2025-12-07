import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class EllipticalDBSCAN:
    def __init__(self, L, min_pts, angle_threshold_degrees, min_branch_size, merge_threshold_degrees=20):
        self.L = L
        self.min_pts = min_pts
        # 1. 초기 분리용 각도 (엄격하게 설정하여 Y자 분기를 확실히 떼어냄)
        self.angle_threshold = np.radians(angle_threshold_degrees)
        self.min_branch_size = min_branch_size
        # 2. 후처리 병합용 각도 (전체 흐름이 비슷하면 끊어진 직선 연결)
        self.merge_threshold = np.radians(merge_threshold_degrees)
        
        self.labels = {}
        self.visited = set()
        self.cluster_map = {} # {child_id: parent_id}
        self.cluster_connections = {} # {child_id: (p_prev, p_curr, p_next)} - 연결 부위 기록
        self.cluster_id_counter = 0

    def get_angle(self, p1, p2):
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    def get_angle_diff(self, theta1, theta2):
        diff = abs(theta1 - theta2)
        if diff > np.pi:
            diff = 2 * np.pi - diff
        return diff

    def check_density(self, p1, p2, data):
        # 최적화를 위해선 KDTree 등이 필요하지만 데모용으로 단순 거리 계산 사용
        count = 0
        for x in data:
            if np.linalg.norm(x - p1) + np.linalg.norm(x - p2) <= self.L:
                count += 1
        return count >= self.min_pts

    def fit(self, data):
        self.labels = {i: -1 for i in range(len(data))}
        self.visited = set()
        self.cluster_id_counter = 0
        self.cluster_map = {}
        self.cluster_connections = {}

        # 1. Main Clustering Loop
        for i in range(len(data)):
            if i in self.visited: continue
            
            # 씨앗 찾기 (거리 L 이내)
            neighbors = [j for j in range(len(data)) 
                         if i != j and np.linalg.norm(data[i] - data[j]) <= self.L]
            
            seed_found = False
            for j in neighbors:
                if j in self.visited: continue
                
                # 밀도 만족 시 군집 시작
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

        # 2. 스마트 병합: 끊어진 일직선 복구 (Collinear Merge)
        self._merge_collinear_segments(data)

        # 3. 노이즈 제거: 미미한 군집 정리
        self._post_process_cleanup()
        
        return np.array([self.labels[i] for i in range(len(data))])

    def _expand_cluster(self, data, queue, current_cluster_id):
        q = deque(queue)
        while q:
            prev_idx, curr_idx = q.popleft()
            p_prev, p_curr = data[prev_idx], data[curr_idx]
            base_angle = self.get_angle(p_prev, p_curr)

            # 후보 탐색
            candidates = [k for k in range(len(data)) 
                          if k != curr_idx and k != prev_idx and k not in self.visited 
                          and np.linalg.norm(data[k] - p_curr) <= self.L]

            for next_idx in candidates:
                p_next = data[next_idx]
                if not self.check_density(p_curr, p_next, data): continue

                new_angle = self.get_angle(p_curr, p_next)
                angle_diff = self.get_angle_diff(base_angle, new_angle)

                if angle_diff <= self.angle_threshold:
                    # 방향 일치 -> 확장
                    self.labels[next_idx] = current_cluster_id
                    self.visited.add(next_idx)
                    q.append((curr_idx, next_idx))
                else:
                    # 방향 불일치 -> 분기 (Branching)
                    self.cluster_id_counter += 1
                    new_cluster_id = self.cluster_id_counter
                    self.labels[next_idx] = new_cluster_id
                    self.visited.add(next_idx)
                    
                    # 관계 및 연결 부위 기록
                    self.cluster_map[new_cluster_id] = current_cluster_id
                    self.cluster_connections[new_cluster_id] = (prev_idx, curr_idx, next_idx)
                    
                    self._expand_cluster(data, [(curr_idx, next_idx)], new_cluster_id)

    def _get_cluster_average_vector(self, data, cluster_id, start_node, num_points=5):
        """군집의 시작/끝 부분에서 국소적인 흐름(벡터)을 추출"""
        points_indices = [i for i, l in self.labels.items() if l == cluster_id]
        if len(points_indices) < 2: return None

        # start_node와 가까운 점들만 추출하여 방향 계산
        nearby_indices = sorted(points_indices, key=lambda idx: np.linalg.norm(data[idx] - data[start_node]))[:num_points]
        
        if len(nearby_indices) < 2: return None
        
        p_start = data[start_node]
        p_end = data[nearby_indices[-1]] # 가장 먼 점
        
        vec = p_end - p_start
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    def _merge_collinear_segments(self, data):
        """분기되었지만 사실상 같은 방향인 군집들을 다시 병합"""
        merge_list = []
        
        for child_id, (p_prev, p_curr, p_next) in self.cluster_connections.items():
            parent_id = self.cluster_map.get(child_id)
            if parent_id is None: continue

            # 1. 부모 군집 방향 (p_curr 기준 안쪽)
            vec_parent = self._get_cluster_average_vector(data, parent_id, start_node=p_curr)
            if vec_parent is not None: vec_parent = -vec_parent # 방향 보정

            # 2. 자식 군집 방향 (p_next 기준 바깥쪽)
            vec_child = self._get_cluster_average_vector(data, child_id, start_node=p_next)

            if vec_parent is None or vec_child is None: continue

            # 3. 각도 비교
            cos_sim = np.dot(vec_parent, vec_child)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            angle = np.arccos(cos_sim)

            if angle <= self.merge_threshold:
                merge_list.append((child_id, parent_id))
        
        # 병합 실행 (간단화된 버전)
        for child, parent in merge_list:
            # 부모 갱신 (1단계)
            real_parent = self.cluster_map.get(parent, parent)
            if real_parent not in self.cluster_connections and real_parent != parent:
                 # 부모가 이미 최상위가 아니라면 최상위 찾기 시도 (생략 가능)
                 pass

            # 라벨 업데이트
            print(f"Merging Collinear: Cluster {child} -> {real_parent}")
            for idx, label in self.labels.items():
                if label == child:
                    self.labels[idx] = real_parent
            
            # 족보 정리
            if child in self.cluster_map:
                del self.cluster_map[child]

    def _post_process_cleanup(self):
        """미미한 크기의 군집은 부모로 환원"""
        counts = {}
        for label in self.labels.values():
            if label == -1: continue
            counts[label] = counts.get(label, 0) + 1
        
        merged = True
        while merged:
            merged = False
            for child, parent in list(self.cluster_map.items()):
                if child in counts and counts[child] < self.min_branch_size:
                    if parent not in counts: continue # 부모가 이미 사라짐
                    
                    # 병합
                    for idx, label in self.labels.items():
                        if label == child: self.labels[idx] = parent
                    
                    counts[parent] += counts[child]
                    del counts[child]
                    del self.cluster_map[child]
                    
                    # 입양 (Grandparent update)
                    for sub_c, sub_p in list(self.cluster_map.items()):
                        if sub_p == child: self.cluster_map[sub_c] = parent
                    
                    merged = True

# =========================================================
# 실행 부분 (Execution Part)
# =========================================================
if __name__ == "__main__":
    # 1. 데이터 생성 (Y자 분기 + 노이즈)
    np.random.seed(42)

    # A. 메인 도로 (직선, 약간의 곡률 추가)
    t = np.linspace(0, 10, 60)
    x1 = t
    y1 = 0.5 * np.sin(t * 0.5) # 완만한 곡선
    road1 = np.column_stack([x1, y1])

    # B. 분기 도로 1 (위로 30도) - 메인 도로 끝에서 분기
    x2 = np.linspace(10, 15, 40)
    y2 = y1[-1] + (x2 - 10) * np.tan(np.radians(30))
    road2 = np.column_stack([x2, y2])

    # C. 분기 도로 2 (아래로 45도) - 메인 도로 끝에서 분기
    x3 = np.linspace(10, 15, 40)
    y3 = y1[-1] + (x3 - 10) * np.tan(np.radians(-45))
    road3 = np.column_stack([x3, y3])

    # D. 노이즈 (랜덤 점)
    noise = np.random.uniform(low=[0, -5], high=[15, 5], size=(50, 2))

    # 데이터 합치기
    X = np.vstack([road1, road2, road3, noise])
    # Jitter (불규칙성) 추가
    X += np.random.normal(0, 0.05, X.shape)

    # 2. 모델 설정 및 실행
    # - angle_threshold=20: 20도 이상 꺾이면 일단 분리 (엄격함 -> Y자 분기 구분)
    # - merge_threshold=25: 분리되었더라도 전체 흐름이 25도 이내면 다시 병합 (유연함 -> 일직선 복구)
    model = EllipticalDBSCAN(L=1.5, min_pts=4, 
                             angle_threshold_degrees=20, 
                             min_branch_size=5,
                             merge_threshold_degrees=25)
    
    print("Fitting model...")
    labels = model.fit(X)
    print("Done.")

    # 3. 결과 시각화
    plt.figure(figsize=(12, 7))

    unique_labels = set(labels)
    # 색상 맵 생성 (노이즈 제외)
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 노이즈 처리
            col = 'k'
            label_name = "Noise"
            marker = 'x'
            size = 30
            alpha = 0.3
        else:
            label_name = f"Cluster {k}"
            marker = 'o'
            size = 50
            alpha = 1.0

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=label_name, s=size, marker=marker, edgecolor='k', alpha=alpha)

    plt.title('Slope-Sensitive Elliptical DBSCAN\n(Strict Separation + Smart Merge)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') # 비율 유지
    plt.show()