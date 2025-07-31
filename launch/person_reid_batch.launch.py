# from launch import LaunchDescription
# from launch_ros.actions import Node

# def generate_launch_description():
#     return LaunchDescription([
#         Node(
#             package='reid_tracker',
#             executable='person_reid_node',
#             name='person_reid_node',
#             parameters=[{"similarity_threshold": 0.88}],
#             output='screen'
#         )
#     ])
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 宣告 image_topic 參數，並設定預設值與說明
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw', #'/camera/image_raw',
        description='要訂閱的影像 topic'
    )
    # 2. 讀取這個參數
    image_topic = LaunchConfiguration('image_topic')

    return LaunchDescription([
        # 將參數宣告加到 LaunchDescription
        image_topic_arg,

        # 節點設定，把 image_topic 作為參數傳入
        Node(
            package='reid_tracker',
            executable='person_reid_batch_node',
            name='person_reid_batch_node',
            parameters=[
                {"similarity_threshold": 0.83},
                {"image_topic": image_topic}
            ],
            output='screen'
        )
    ])

