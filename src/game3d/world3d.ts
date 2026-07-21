import type { GameSceneId } from '../game/types';
import type {
  Game3DSceneDefinition,
  NavigationObstacle3D,
  NavigationPath3D,
  NavigationZone3D,
} from './types3d';

/**
 * Open voxel world (Minicraft-like exploration scale) with a research-station hub.
 * Hub remains densest; outer map is procedural terrain + outposts.
 */
export const campusPathNetwork3D: NavigationPath3D[] = [
  // Hub spines
  path('pier-spine', [[0, 28], [0, 18], [0, 8], [0, 0], [0, -12], [0, -24]], 4.2),
  path('east-road', [[0, 8], [12, 8], [24, 8], [36, 4], [48, -8]], 3.4),
  path('west-road', [[0, 8], [-12, 8], [-24, 6], [-36, -4], [-48, -16]], 3.4),
  path('north-ridge', [[0, -24], [0, -36], [0, -48], [12, -52], [-12, -52]], 3.2),
  path('south-beach', [[0, 28], [-16, 36], [16, 36], [0, 44]], 3.6),
  path('academy-spur', [[0, 4], [-6, 4], [-10, 4]], 2.8),
  path('plaza-spur', [[0, 8], [6, 8], [11, 8]], 2.8),
  path('lab-spur', [[0, -8], [-6, -8], [-10, -8]], 2.8),
  path('compute-spur', [[0, -6], [6, -6], [11, -6]], 2.8),
  path('outpost-ne', [[24, 8], [34, -10], [38, -22]], 2.6),
  path('outpost-nw', [[-24, 6], [-34, -12], [-40, -28]], 2.6),
  path('outpost-se', [[16, 36], [28, 34], [38, 30]], 2.6),
  path('outpost-sw', [[-16, 36], [-30, 32], [-40, 24]], 2.6),
  path('far-north', [[0, -48], [0, -70], [-20, -80], [20, -80]], 3.0),
  path('far-east', [[48, -8], [70, -10], [80, 10]], 3.0),
  path('far-west', [[-48, -16], [-70, -12], [-80, 8]], 3.0),
  path('cave-spur-n', [[0, -36], [-8, -48], [-8, -52]], 2.4),
  path('cave-spur-ne', [[24, -12], [34, -22], [42, -38]], 2.4),
  path('cave-spur-nw', [[-24, -12], [-36, -28], [-46, -34]], 2.4),
];

const campusWalkableZones: NavigationZone3D[] = [
  // Nearly whole open map is walkable; zones help pathfinding density near hubs.
  zone(0, 0, 96, 96),
  zone(0, 24, 8, 6),
  zone(0, 8, 10, 8),
  zone(-10, 4, 5, 4),
  zone(11, 8, 5, 4),
  zone(-10, -8, 5, 4),
  zone(11, -6, 5, 4),
  zone(0, -18, 6, 5),
  zone(-36, -28, 6, 6),
  zone(34, -22, 6, 6),
  zone(-40, 24, 6, 6),
  zone(38, 30, 6, 6),
  zone(0, -48, 8, 6),
];

export const gameScenes3D: Record<Game3DSceneDefinition['id'], Game3DSceneDefinition> = {
  campus: {
    id: 'campus',
    name: '体素观测大陆',
    subtitle: '开放大陆 · 草甸/海岸/山脊 · 北脊可进洞穴',
    spawn: [0, 1.15, 22],
    bounds: { minX: -108, maxX: 108, minZ: -108, maxZ: 108 },
    navigation: {
      cellSize: 1.4,
      shape: 'rectangle',
      padding: 1.2,
      paths: campusPathNetwork3D,
      zones: campusWalkableZones,
      obstacles: [
        obstacle(-10, 4, 5.6, 4.8),
        obstacle(11, 8, 5.2, 4.6),
        obstacle(-10, -8, 5.4, 5.0),
        obstacle(11, -6, 5.6, 5.2),
        obstacle(0, -18, 6.2, 4.4),
        obstacle(0, -2, 4.8, 4.8),
        // outpost footprints (approx)
        obstacle(-36, -28, 3.2, 3.2),
        obstacle(34, -22, 3.2, 3.2),
        obstacle(-40, 24, 3.2, 3.2),
        obstacle(38, 30, 3.2, 3.2),
        obstacle(0, -48, 3.4, 3.4),
      ],
    },
    interactions: [
      {
        id: 'guide',
        kind: 'agent',
        label: '与值班员交谈',
        position: [1.4, 0, 15.5],
        agentId: 'campus-guide',
        radius: 2.5,
      },
      {
        id: 'academy-door',
        kind: 'portal',
        label: '进入方法讲习室',
        position: [-10, 0, 4],
        targetScene: 'academy',
        radius: 1.4,
      },
      {
        id: 'research-door',
        kind: 'portal',
        label: '进入证据实验室',
        position: [-10, 0, -8],
        targetScene: 'research',
        radius: 1.4,
      },
      {
        id: 'compute-door',
        kind: 'portal',
        label: '进入崖侧机房',
        position: [11, 0, -6],
        targetScene: 'compute',
        radius: 1.4,
      },
      {
        id: 'paper-door',
        kind: 'portal',
        label: '进入评议厅',
        position: [0, 0, -18],
        targetScene: 'paper',
        radius: 1.4,
      },
      {
        id: 'plaza-door',
        kind: 'portal',
        label: '进入公告廊',
        position: [11, 0, 8],
        targetScene: 'plaza',
        radius: 1.4,
      },
      landmark(
        'open-data-pavilion',
        '查看栈桥数据柜',
        '开放数据不是把文件放出来就结束。记录许可证、采集过程、字段语义与已知缺口。',
        [-3.2, 0, 20],
      ),
      landmark(
        'replication-garden',
        '查看复现对照架',
        '先冻结代码、数据、随机种子和环境，再讨论结果差异。',
        [3.4, 0, 12],
      ),
      landmark(
        'ethics-overlook',
        '查看伦理告示板',
        '在实验开始前写下潜在伤害、数据边界和退出条件。',
        [-4.2, 0, 0],
      ),
      landmark(
        'field-notes-deck',
        '使用观测记录台',
        '把观察与解释分开记录：先写发生了什么，再写你认为为什么发生。',
        [4.5, 0, -10],
      ),
      landmark(
        'negative-results-bell',
        '查看负结果通告',
        '负结果若能排除一个合理假设，同样缩小了未知空间。',
        [-3.5, 0, -14],
      ),
      landmark(
        'collaboration-hearth',
        '阅读值班交接',
        '有效协作从明确问题所有权、接口、证据标准和决策记录开始。',
        [3.2, 0, 6],
      ),
      // Outer open-world discovery nodes
      landmark(
        'ruin-ne',
        '勘查东北前哨废墟',
        '旧实验站留下的日志残页：有人改过切分协议却没写进海报。',
        [34, 0, -22],
      ),
      landmark(
        'ruin-nw',
        '勘查西北苔原桩站',
        '苔原上的传感器阵列仍在跳变——可能是噪声，也可能是泄漏信号。',
        [-36, 0, -28],
      ),
      landmark(
        'beach-cache',
        '翻找南岸补给箱',
        '补给箱里塞着打印失败的基准分数表，墨迹被潮水晕开。',
        [20, 0, 44],
      ),
      landmark(
        'ridge-beacon',
        '检查北脊信标',
        '信标闪着暖黄光：远距协作约定写在底座铜牌上。',
        [0, 0, -48],
      ),
      landmark(
        'cave-north',
        '进入北脊主洞',
        '洞口被高度场压成缓坡，隧道里有发光晶体——也许是旧实验的频谱探针。',
        [-8, 0, -52],
      ),
      landmark(
        'cave-ne',
        '勘查东北矿洞',
        '岩壁上刻着被划掉的 seed 表，像有人想销毁复现证据。',
        [42, 0, -38],
      ),
      landmark(
        'cave-nw',
        '进入西北苔洞',
        '洞内潮湿，传感器线缆半埋在石缝里仍在闪。',
        [-46, 0, -34],
      ),
      landmark(
        'cave-mid',
        '顺着脊线裂隙下探',
        '裂隙尽头的晶体旁贴着一张「勿公开切分」的便签。',
        [18, 0, -58],
      ),
    ],
  },
  academy: {
    id: 'academy',
    name: '方法讲习室',
    subtitle: '与导师对质方法：对照、泄漏与可反驳假设',
    spawn: [0, 1.15, 7.1],
    bounds: { minX: -11, maxX: 11, minZ: -9, maxZ: 9 },
    navigation: {
      cellSize: 0.45,
      shape: 'rectangle',
      padding: 0.65,
      obstacles: [
        obstacle(-6.5, -4.5, 5.1, 2),
        obstacle(6.5, -4.5, 5.1, 2),
        obstacle(-6.7, 2.3, 4.2, 2.2),
        obstacle(6.7, 2.3, 4.2, 2.2),
      ],
    },
    interactions: [
      {
        id: 'academy-mentor',
        kind: 'agent',
        label: '与苏老师讨论科研方法',
        position: [0, 0, -2.8],
        agentId: 'research-mentor',
        radius: 2.8,
      },
      {
        id: 'academy-exit',
        kind: 'portal',
        label: '返回临海观测站',
        position: [0, 0, 8.15],
        targetScene: 'campus',
        radius: 1.3,
      },
    ],
  },
  research: {
    id: 'research',
    name: '模型与系统实验室',
    subtitle: '审计数据与基线，冻结研究问题和可反驳假设',
    spawn: [0, 1.15, 7.2],
    bounds: { minX: -12, maxX: 12, minZ: -9, maxZ: 9 },
    navigation: {
      cellSize: 0.45,
      shape: 'rectangle',
      padding: 0.65,
      obstacles: [
        obstacle(-6.7, -4.9, 4.6, 1.8),
        obstacle(5.6, -4.6, 4.2, 2.2),
        obstacle(5.8, 1.4, 4.5, 1.6),
        obstacle(-4.2, 2.7, 3.8, 2.4),
      ],
    },
    interactions: [
      {
        id: 'methods-mentor',
        kind: 'agent',
        label: '与研究方法导师周岚交谈',
        position: [-3.8, 0, -2.4],
        agentId: 'methods-mentor',
        radius: 2.5,
      },
      {
        id: 'project-terminal',
        kind: 'workspace',
        label: '使用项目与数据终端',
        position: [5.6, 0, -4.6],
        workspaceZone: 'observatory',
        radius: 2.3,
      },
      {
        id: 'hypothesis-board',
        kind: 'workspace',
        label: '使用研究假设白板',
        position: [5.8, 0, 1.4],
        workspaceZone: 'lab',
        radius: 2.3,
      },
      {
        id: 'research-exit',
        kind: 'portal',
        label: '返回临海观测站',
        position: [0, 0, 8.1],
        targetScene: 'campus',
        radius: 1.35,
      },
    ],
  },
  compute: {
    id: 'compute',
    name: '算力工坊',
    subtitle: '启动固定实验，观察逐样本指标并处理失败恢复',
    spawn: [0, 1.15, 7.2],
    bounds: { minX: -12, maxX: 12, minZ: -9, maxZ: 9 },
    navigation: {
      cellSize: 0.45,
      shape: 'rectangle',
      padding: 0.65,
      obstacles: [
        obstacle(-7.3, -3.4, 2.4, 8.4),
        obstacle(0, -3.4, 2.4, 8.4),
        obstacle(7.3, -3.4, 2.4, 8.4),
        obstacle(5.4, 3.2, 4.2, 2),
      ],
    },
    interactions: [
      {
        id: 'compute-agent',
        kind: 'agent',
        label: '与实验工程师程野交谈',
        position: [-3.9, 0, 2.7],
        agentId: 'experiment-engineer',
        radius: 2.5,
      },
      {
        id: 'cluster-console',
        kind: 'workspace',
        label: '操作离线实验集群',
        position: [5.4, 0, 3.2],
        workspaceZone: 'compute',
        radius: 2.3,
      },
      {
        id: 'compute-exit',
        kind: 'portal',
        label: '返回临海观测站',
        position: [0, 0, 8.1],
        targetScene: 'campus',
        radius: 1.35,
      },
    ],
  },
  paper: {
    id: 'paper',
    name: '论文与评审工坊',
    subtitle: '让证据经受审稿质询，再组织分析、复现与写作',
    spawn: [0, 1.15, 7.2],
    bounds: { minX: -12, maxX: 12, minZ: -9, maxZ: 9 },
    navigation: {
      cellSize: 0.45,
      shape: 'rectangle',
      padding: 0.65,
      obstacles: [
        obstacle(-6.4, -4.5, 4.8, 1.8),
        obstacle(5.8, -4.4, 4.8, 2.4),
        obstacle(-5.5, 2.6, 4.4, 2),
        obstacle(5.5, 2.6, 4.4, 2),
      ],
    },
    interactions: [
      {
        id: 'reviewer',
        kind: 'agent',
        label: '接受审稿人顾言的质询',
        position: [-3.9, 0, -1.8],
        agentId: 'reviewer',
        radius: 2.5,
      },
      {
        id: 'manuscript-desk',
        kind: 'workspace',
        label: '打开分析与复现稿件',
        position: [5.8, 0, -4.4],
        workspaceZone: 'paper',
        radius: 2.3,
      },
      {
        id: 'paper-exit',
        kind: 'portal',
        label: '返回临海观测站',
        position: [0, 0, 8.1],
        targetScene: 'campus',
        radius: 1.35,
      },
    ],
  },
  plaza: {
    id: 'plaza',
    name: '学术交流广场',
    subtitle: '展示可验证成果，与其他研究者交换具体反馈',
    spawn: [0, 1.15, 8.1],
    bounds: { minX: -13, maxX: 13, minZ: -10, maxZ: 10 },
    navigation: {
      cellSize: 0.5,
      shape: 'ellipse',
      padding: 0.75,
      obstacles: [
        obstacle(-7, -3.6, 3.8, 1.3),
        obstacle(5.4, -3.8, 4.4, 1.6),
        obstacle(-5.8, 3, 3.2, 1.4),
        obstacle(5.8, 3, 3.2, 1.4),
      ],
    },
    interactions: [
      {
        id: 'community-host',
        kind: 'agent',
        label: '与学术社区导师许禾交谈',
        position: [-3.7, 0, -1.2],
        agentId: 'community-host',
        radius: 2.5,
      },
      {
        id: 'poster-board',
        kind: 'workspace',
        label: '查看验证研究海报板',
        position: [5.4, 0, -3.8],
        workspaceZone: 'plaza',
        radius: 2.3,
      },
      {
        id: 'plaza-exit',
        kind: 'portal',
        label: '返回临海观测站',
        position: [0, 0, 9],
        targetScene: 'campus',
        radius: 1.35,
      },
    ],
  },
};

export function isThreeSceneId(value: string): value is Game3DSceneDefinition['id'] {
  return value in gameScenes3D;
}

export const gameSceneIds3D = Object.keys(gameScenes3D) as GameSceneId[];

function obstacle(x: number, z: number, width: number, depth: number): NavigationObstacle3D {
  return {
    minX: x - width / 2,
    maxX: x + width / 2,
    minZ: z - depth / 2,
    maxZ: z + depth / 2,
  };
}

function path(id: string, points: Array<[number, number]>, width: number): NavigationPath3D {
  return { id, points, width };
}

function zone(x: number, z: number, radiusX: number, radiusZ: number): NavigationZone3D {
  return { x, z, radiusX, radiusZ };
}

function landmark(
  id: string,
  label: string,
  note: string,
  position: [number, number, number],
) {
  return {
    id,
    kind: 'landmark' as const,
    label,
    title: label.replace(/^查看|^观察|^使用|^敲响|^阅读/, ''),
    note,
    position,
    radius: 2.3,
  };
}
