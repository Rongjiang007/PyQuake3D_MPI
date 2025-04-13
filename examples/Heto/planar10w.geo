// 参数定义
angle = 25; // 倾角 (度)
dx = 50;    // 矩形宽度
Len = 30;   // 矩形长度
lc = 0.12;     // 单元大小

// 定义点
Point(1) = {-dx/2, 0, 0, lc};
Point(2) = {dx/2, 0, 0, lc};
Point(3) = {dx/2, 0, -Len, lc};
Point(4) = {-dx/2, 0, -Len, lc};
//Point(3) = {dx, Len * Cos(angle * Pi / 180), -Len * Sin(angle * Pi / 180), lc};
//Point(4) = {0, Len * Cos(angle * Pi / 180), -Len * Sin(angle * Pi / 180), lc};

// 定义线段
Line(1) = {1, 2}; // A -> B
Line(2) = {2, 3}; // B -> C
Line(3) = {3, 4}; // C -> D
Line(4) = {4, 1}; // D -> A

// 定义面
Line Loop(1) = {1, 2, 3, 4}; // 面 A-B-C-D
Plane Surface(1) = {1};

// 网格设置
Mesh.Algorithm = 6;          // 使用指定算法生成网格 (例如 Delaunay)
Mesh.ElementSizeFactor = 1.0; // 全局网格因子 (可选)

