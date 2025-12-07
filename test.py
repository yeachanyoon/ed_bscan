import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class EllipticalDBSCAN:
    def __init__(self, L, min_pts, angle_threshold_degrees, min_branch_size):
        self.L = L  # 타원의 장축 길이 (탐색 범위)
        self.min_pts = min_pts # 타원 내 최소 포인트 수
        self.angle_threshold = np.radians(angle_threshold_degrees) # 허용 각도 (라디안 변환)
        self.min_branch_size = min_branch_size # 환원 기준 크기
        
        self.labels = {} # {point_index: cluster_id}
        self.visited = set() # 방문한 점 인덱스
        self.cluster_map = {} # {cluster_id: parent_id} (족보)
        self.cluster_id_counter = 0

    def get_angle(self, p1, p2):
        """두 점 사이의 각도 (라디안)"""
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    def get_angle_diff(self, theta1, theta2):
        """두 각도의 최소 차이 (0 ~ pi)"""
        diff = abs(theta1 - theta2)
        if diff > np.pi:
            diff = 2 * np.pi - diff
        return diff

    def check_density(self, p1, p2, data):
        """
        두 초점(p1, p2)으로 만든 타원 내부에 min_pts 이상 존재하는지 확인
        조건: dist(x, p1) + dist(x, p2) <= L
        """
        count = 0
        # 최적화를 위해선 KDTree 등을 써야 하지만, 데모용으로 단순 반복 사용
        for x in data:
            if np.linalg.norm(x - p1) + np.linalg.norm(x - p2) <= self.L:
                count += 1
        return count >= self.min_pts

    def fit(self, data):
        self.labels = {i: -1 for i in range(len(data))} # -1은 노이즈 초기화
        self.visited = set()
        self.cluster_id_counter = 0

        # 모든 점을 순회하며 씨앗(Seed) 탐색
        for i in range(len(data)):
            if i in self.visited:
                continue

            # 초기 파트너 찾기 (거리 L 이내의 이웃)
            initial_neighbors = []
            for j in range(len(data)):
                if i == j: continue
                if np.linalg.norm(data[i] - data[j]) <= self.L:
                    initial_neighbors.append(j)
            
            # 유효한 시작점이 되는지 확인
            seed_found = False
            for j in initial_neighbors:
                if j in self.visited: continue
                
                # 초기 타원 밀도 확인
                if self.check_density(data[i], data[j], data):
                    # 새로운 군집 시작
                    self.cluster_id_counter += 1
                    current_cluster = self.cluster_id_counter
                    
                    self.labels[i] = current_cluster
                    self.labels[j] = current_cluster
                    self.visited.add(i)
                    self.visited.add(j)
                    
                    # 확장 시작 (큐에 직전점, 현재점 정보 저장)
                    self._expand_cluster(data, [(i, j)], current_cluster)
                    seed_found = True
                    break # 하나의 씨앗으로 확장을 시작했으면 다음 점으로 넘어감
            
            if not seed_found:
                self.labels[i] = -1 # 노이즈로 유지

        # 후처리: 미미한 군집 환원
        self._post_process_merge()
        return np.array([self.labels[i] for i in range(len(data))])

    def _expand_cluster(self, data, queue, current_cluster_id):
        # BFS 방식 탐색
        q = deque(queue) # (prev_idx, curr_idx)

        while q:
            prev_idx, curr_idx = q.popleft()
            
            p_prev = data[prev_idx]
            p_curr = data[curr_idx]
            
            # 1. 기준 각도 (이전 -> 현재)
            base_angle = self.get_angle(p_prev, p_curr)

            # 2. 다음 후보 탐색 (현재점에서 L 거리 이내, 미방문)
            # 주의: 방문했던 점이라도 '분기' 처리를 위해 체크할 필요가 있을 수 있으나,
            # 여기서는 간단히 미방문 점만 대상으로 함
            candidates = []
            for k in range(len(data)):
                if k == curr_idx or k == prev_idx: continue
                if k in self.visited: continue
                
                # 거리 1차 필터링
                if np.linalg.norm(data[k] - p_curr) <= self.L:
                    candidates.append(k)

            for next_idx in candidates:
                p_next = data[next_idx]

                # A. 밀도 체크 (새로운 타원 형성 가능성)
                if not self.check_density(p_curr, p_next, data):
                    continue

                # B. 각도 체크
                new_angle = self.get_angle(p_curr, p_next)
                angle_diff = self.get_angle_diff(base_angle, new_angle)

                if angle_diff <= self.angle_threshold:
                    # [Case 1] 방향 일치 -> 현재 군집 확장
                    self.labels[next_idx] = current_cluster_id
                    self.visited.add(next_idx)
                    q.append((curr_idx, next_idx))
                
                else:
                    # [Case 2] 방향 불일치 -> 분기 (Branching)
                    # 새로운 군집 생성
                    self.cluster_id_counter += 1
                    new_cluster_id = self.cluster_id_counter
                    
                    self.labels[next_idx] = new_cluster_id
                    self.visited.add(next_idx)
                    
                    # 족보 기록 (자식 -> 부모)
                    self.cluster_map[new_cluster_id] = current_cluster_id
                    
                    # 재귀적 호출 대신, 큐에 넣지 않고 별도 처리하거나
                    # 여기서는 단순화를 위해 현재 큐 루프와 별개로 재귀 호출과 유사하게 처리
                    # 하지만 BFS 구조상 큐에 넣되 ID를 다르게 관리해야 함.
                    # 구조 단순화를 위해 여기서는 '새로운 큐'를 만들어서 expand를 호출
                    self._expand_cluster(data, [(curr_idx, next_idx)], new_cluster_id)

    def _post_process_merge(self):
        """크기가 작은 파생 군집을 부모 군집으로 병합 (오류 수정판)"""
        
        # 1. 초기 군집 크기 계산
        counts = {}
        for label in self.labels.values():
            if label == -1: continue
            counts[label] = counts.get(label, 0) + 1
            
        merged_something = True
        while merged_something:
            merged_something = False
            
            # cluster_map을 순회하며 병합 대상 탐색
            # list()로 감싸서 반복문 중 딕셔너리 변경 허용
            for child_id, parent_id in list(self.cluster_map.items()):
                
                # (1) 자식 군집이 존재하고, 크기가 기준보다 작은가?
                if child_id in counts and counts[child_id] < self.min_branch_size:
                    
                    # (2) 부모 군집이 counts에 존재하는지 확인 (KeyError 방지)
                    if parent_id not in counts:
                        # 부모가 이미 다른 곳에 병합되어 사라진 경우입니다.
                        # 이 경우, 현재 map 정보가 낡은 것이므로 일단 넘어갑니다.
                        # (아래의 '입양 로직'에 의해 다음 턴에 부모가 갱신됩니다)
                        continue

                    # (3) 병합 수행
                    # print(f"Merge: Cluster {child_id} -> Parent {parent_id}")
                    
                    # 라벨 변경: 자식 ID를 가진 모든 점을 부모 ID로 변경
                    for idx, label in self.labels.items():
                        if label == child_id:
                            self.labels[idx] = parent_id
                    
                    # 크기 합치기
                    counts[parent_id] += counts[child_id]
                    del counts[child_id]
                    
                    # 맵에서 현재 자식 관계 삭제
                    del self.cluster_map[child_id]
                    
                    # (4) ★중요: 입양 로직 (Grandparent Update)★
                    # 만약 "현재 병합된 child_id"를 부모로 섬기던 다른 군집들이 있다면,
                    # 그들의 부모를 "현재의 parent_id"로 갱신해줘야 함.
                    # 예: A <- B <- C 상황에서 B가 A로 병합되면, C의 부모를 A로 바꿔줌.
                    for sub_child, sub_parent in list(self.cluster_map.items()):
                        if sub_parent == child_id:
                            self.cluster_map[sub_child] = parent_id
                    
                    merged_something = True

# ---------------------------------------------------------
# 1. 데이터 생성 (Y자 분기 + 노이즈)
# ---------------------------------------------------------
np.random.seed(42)

# 메인 도로 (직선)
x1 = np.linspace(0, 10, 50)
y1 = np.zeros_like(x1)
road1 = np.column_stack([x1, y1])

# 분기 도로 1 (위로 30도)
x2 = np.linspace(10, 15, 30)
y2 = (x2 - 10) * np.tan(np.radians(30))
road2 = np.column_stack([x2, y2])

# 분기 도로 2 (아래로 45도) - 메인 도로 끝에서 분기
x3 = np.linspace(10, 15, 30)
y3 = (x3 - 10) * np.tan(np.radians(-45))
road3 = np.column_stack([x3, y3])

# 노이즈 (랜덤 점)
noise = np.random.uniform(low=[0, -5], high=[15, 5], size=(30, 2))

# 데이터 합치기
X = np.vstack([road1, road2, road3, noise])
# 약간의 jitter 추가 (완벽한 직선 방지)
X += np.random.normal(0, 0.1, X.shape)

# ---------------------------------------------------------
# 2. 모델 실행
# ---------------------------------------------------------
# L: 탐색 길이, min_pts: 밀도, angle: 40도 이내면 같은 길, min_branch: 5개 이하면 노이즈 처리
model = EllipticalDBSCAN(L=2, min_pts=5, angle_threshold_degrees=35, min_branch_size=10)
labels = model.fit(X)

# ---------------------------------------------------------
# 3. 결과 시각화
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # 노이즈는 검은색
        col = 'k'
        label_name = "Noise"
        marker = 'x'
        size = 20
    else:
        label_name = f"Cluster {k}"
        marker = 'o'
        size = 60

    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=label_name, s=size, marker=marker, edgecolor='k')

plt.title('Proposed Algorithm: Slope-Sensitive Elliptical DBSCAN')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()