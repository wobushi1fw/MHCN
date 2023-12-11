import tkinter as tk
import json
import os

class RecommenderApp:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.recListTagWithValue = self.load_rec_list('recListTagWithValue.json')
        self.recListArtistWithName = self.load_rec_list('recListArtistWithName.json')

        # 创建 Tkinter 窗口
        self.window = tk.Tk()
        self.window.title("用户推荐信息")

        # 用户ID输入框
        self.user_entry_label = tk.Label(self.window, text="请输入用户ID:")
        self.user_entry_label.pack()
        self.user_entry = tk.Entry(self.window)
        self.user_entry.pack()

        # 查询按钮
        self.query_button = tk.Button(self.window, text="个性化推荐", command=self.show_recommendations)
        self.query_button.pack()

        # 社区推荐按钮
        self.community_button = tk.Button(self.window, text="社区推荐", command=self.show_community_recommendation_window)
        self.community_button.pack()

        # 退出按钮
        self.quit_button = tk.Button(self.window, text="退出", command=self.window.destroy)
        self.quit_button.pack()

        # Center the window on the screen
        self.center_window()

    def center_window(self, window=None):
        # 获取屏幕宽度和高度
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # 使用默认窗口或传递的窗口
        target_window = window if window else self.window

        # 设置窗口的尺寸和位置
        target_window.geometry(f'{screen_width}x{screen_height}+0+0')

    def load_rec_list(self, filename):
        # 构建文件路径
        file_path = os.path.join(self.folder_path, filename)

        # 加载 JSON 文件
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                recList = json.load(file)
            return recList
        else:
            print(f"File {file_path} does not exist.")
            return {}

    def show_recommendations(self):
        user_id = self.user_entry.get()

        # 检查用户是否存在
        if user_id not in self.recListTagWithValue or user_id not in self.recListArtistWithName:
            self.show_message(f"缺乏用户 {user_id} 的信息，无法做推荐。")
            return

        user_tag_recommendations = self.recListTagWithValue[user_id]
        user_artist_recommendations = self.recListArtistWithName[user_id]

        # 在窗口中显示推荐信息
        result_text = f"{user_id} 的推荐Tag: "
        result_text += ", ".join([f"{tag_item[1]}" for tag_item in user_tag_recommendations])

        result_text += f"\n{user_id} 的推荐Artist: "
        result_text += ", ".join([f"{artist_item[1]}" for artist_item in user_artist_recommendations])

        self.show_message(result_text)

    def show_community_recommendation_window(self):
        community_recommendation_window = tk.Toplevel(self.window)
        community_recommendation_window.title("社区推荐")

        # Center the community recommendation window on the screen
        self.center_window(community_recommendation_window)

        # 输入框
        input_label = tk.Label(community_recommendation_window, text="请输入Artist的name或Tag的Value:")
        input_label.pack()
        input_entry = tk.Entry(community_recommendation_window)
        input_entry.pack()

        # 查询按钮
        query_button = tk.Button(community_recommendation_window, text="查询", command=lambda: self.show_cluster_window('TagClusterWithValue.txt', input_entry))
        query_button.pack()

        # 返回按钮
        return_button = tk.Button(community_recommendation_window, text="返回", command=community_recommendation_window.destroy)
        return_button.pack()

    def show_cluster_window(self, cluster_filename, text_widget):
        input_value = text_widget.get()

        # 确定小部件的类型并相应地处理
        if isinstance(text_widget, tk.Entry):
            self.show_cluster_window_entry(input_value, cluster_filename, text_widget)

    def show_cluster_window_entry(self, input_value, cluster_filename, text_widget):
        # 适用于 Entry 组件的处理方式
        cluster_data = self.load_cluster_data(cluster_filename)
        matching_clusters = [cluster for cluster in cluster_data if input_value in cluster]

        for idx, cluster in enumerate(matching_clusters):
            cluster_button = tk.Button(text_widget.master, text=f"Cluster {idx + 1}",
                                       command=lambda c=cluster: self.show_detail_window(c, text_widget))
            cluster_button.pack()

    def show_detail_window(self, detail_data, text_widget):
        detail_window = tk.Toplevel(self.window)
        detail_window.title("详细信息")

        # Center the detail window on the screen
        self.center_window(detail_window)

        # 在详细信息窗口中显示具体数据
        detail_text = tk.Text(detail_window, wrap="word")
        detail_text.pack(expand=True, fill="both")
        detail_text.insert("end", f"详细信息： {detail_data}")

        # 添加返回按钮
        return_button = tk.Button(detail_window, text="返回", command=detail_window.destroy)
        return_button.pack()

    def load_cluster_data(self, filename):
        # 构建文件路径
        file_path = os.path.join(self.folder_path, filename)

        # 加载 Cluster 数据
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                clusters = [line.strip().split(', ') for line in file.readlines()]
            return clusters
        else:
            print(f"File {file_path} does not exist.")
            return []

    def show_message(self, message):
        message_window = tk.Toplevel(self.window)
        message_window.title("消息")
        self.center_window(message_window)

        # 显示消息
        message_label = tk.Label(message_window, text=message)
        message_label.pack()

        # 添加确定按钮
        ok_button = tk.Button(message_window, text="确定", command=message_window.destroy)
        ok_button.pack()

    def start(self):
        self.window.mainloop()

# 示例用法
save_self_folder = 'FinalData'  # 请确保文件夹存在
app = RecommenderApp(save_self_folder)
app.start()


