# function to search the path
def a_star_search(grid: list, begin_point: list, target_point: list, cost=1):
    assert ((grid[begin_point[0]][begin_point[1]] != 1) and (grid[target_point[0]][target_point[1]] != 1))

    # the cost map which pushes the path closer to the goal
    heuristic = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            heuristic[i][j] = abs(i - target_point[0]) + abs(j - target_point[1])
            if grid[i][j] == 1:
                heuristic[i][j] = 99  # added extra penalty in the heuristic map

    # the actions we can take
    delta = [[-1, 0],  # go up
             [0, -1],  # go left
             [1, 0],  # go down
             [0, 1]]  # go right

    close_matrix = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]  # the referrence grid
    close_matrix[begin_point[0]][begin_point[1]] = 1
    action_matrix = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]  # the action grid

    x = begin_point[0]
    y = begin_point[1]
    g = 0
    f = g + heuristic[begin_point[0]][begin_point[0]]
    cell = [[f, g, x, y]]

    found = False  # flag that is set when search is complete
    resign = False  # flag set if we can't find expand

    while not found and not resign:
        if len(cell) == 0:
            resign = True
            return None, None
        else:
            cell.sort()  # to choose the least costliest action so as to move closer to the goal
            cell.reverse()
            next = cell.pop()
            x = next[2]
            y = next[3]
            g = next[1]
            f = next[0]

            if x == target_point[0] and y == target_point[1]:
                found = True
            else:
                # delta have four steps
                for i in range(len(delta)):  # to try out different valid actions
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):  # 判断可否通过那个点
                        if close_matrix[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            f2 = g2 + heuristic[x2][y2]
                            cell.append([f2, g2, x2, y2])
                            close_matrix[x2][y2] = 1
                            action_matrix[x2][y2] = i
    invpath = []
    x = target_point[0]
    y = target_point[1]
    invpath.append([x, y])  # we get the reverse path from here
    while x != begin_point[0] or y != begin_point[1]:
        x2 = x - delta[action_matrix[x][y]][0]
        y2 = y - delta[action_matrix[x][y]][1]
        x = x2
        y = y2
        invpath.append([x, y])

    path = []
    for i in range(len(invpath)):
        path.append(invpath[len(invpath) - 1 - i])
    return path, action_matrix


if __name__ == "__main__":
	# 0为自由通行节点，1为障碍
    grid = [[0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0]]

    begin = [3, 0]  # 3行0列为起始点
    target = [1, 4]  # 1行4列为终点

    a_star_path, action_matrix = a_star_search(grid, begin, target)
    for path in a_star_path:
        print(path)
