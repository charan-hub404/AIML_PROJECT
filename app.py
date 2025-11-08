import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from game_generator import BoardGameGenerator
from ai_player import QLearningPlayer
from game_simulator import GameSimulator
from balance_analyzer import GameBalanceAnalyzer
import time

st.set_page_config(
    page_title="Self-Learning Board Game Designer",
    page_icon="🎲",
    layout="wide"
)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = BoardGameGenerator()
if 'current_game' not in st.session_state:
    st.session_state.current_game = None
if 'ai_players' not in st.session_state:
    st.session_state.ai_players = None
if 'simulator' not in st.session_state:
    st.session_state.simulator = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = GameBalanceAnalyzer()
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = []
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'game_ratings' not in st.session_state:
    st.session_state.game_ratings = {}

# Title and description
st.title("🎲 Self-Learning Board Game Designer")
st.markdown("""
This AI system autonomously generates board game concepts and uses reinforcement learning 
to improve game balance and engagement over time.
""")

# Sidebar controls
st.sidebar.header("🎮 Game Generation")

genre = st.sidebar.selectbox(
    "Select Genre",
    ["Random", "Strategy", "Resource Management", "Racing", "Territory Control"]
)

complexity = st.sidebar.slider(
    "Complexity Level",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

if st.sidebar.button("🎲 Generate New Game", type="primary"):
    genre_param = None if genre == "Random" else genre.lower().replace(" ", "_")
    st.session_state.current_game = st.session_state.generator.generate_game(
        genre=genre_param,
        complexity=complexity
    )
    
    # Initialize AI players
    num_players = st.session_state.current_game['num_players']
    st.session_state.ai_players = [
        QLearningPlayer(i, learning_rate=0.1, discount_factor=0.9, epsilon=0.3)
        for i in range(num_players)
    ]
    
    # Initialize simulator
    st.session_state.simulator = GameSimulator(st.session_state.current_game)
    st.session_state.simulation_results = []
    st.session_state.training_history = []
    
    st.sidebar.success("New game generated!")

st.sidebar.markdown("---")

# Training controls
st.sidebar.header("🤖 AI Training")

num_games = st.sidebar.number_input(
    "Games to Simulate",
    min_value=1,
    max_value=500,
    value=50,
    step=10
)

if st.sidebar.button("▶️ Start Training", disabled=st.session_state.current_game is None):
    if st.session_state.current_game and st.session_state.ai_players:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        for i in range(num_games):
            # Run simulation
            result = st.session_state.simulator.simulate_game(
                st.session_state.ai_players,
                training=True,
                render=False
            )
            st.session_state.simulation_results.append(result)
            
            # Track training progress
            if i % 10 == 0:
                stats = [player.get_stats() for player in st.session_state.ai_players]
                st.session_state.training_history.append({
                    'game_number': len(st.session_state.simulation_results),
                    'players': stats
                })
            
            # Update progress
            progress_bar.progress((i + 1) / num_games)
            status_text.text(f"Training: {i + 1}/{num_games} games")
        
        status_text.text(f"✅ Completed {num_games} games!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

# Main content area
if st.session_state.current_game is None:
    st.info("👈 Generate a new game from the sidebar to get started!")
else:
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Game Design", 
        "🤖 AI Performance", 
        "📊 Balance Analysis",
        "💭 Human Feedback"
    ])
    
    # Tab 1: Game Design
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Game Concept")
            game = st.session_state.current_game
            
            st.markdown(f"### {game['name']}")
            st.markdown(f"**Genre:** {game['genre'].replace('_', ' ').title()}")
            st.markdown(f"**Players:** {game['num_players']}")
            st.markdown(f"**Complexity:** {game['complexity']:.0%}")
            
            st.markdown("---")
            
            st.subheader("📜 Rules & Mechanics")
            st.markdown(f"**Objective:** {game['objective']}")
            
            mechanics = ', '.join([m.replace('_', ' ').title() for m in game['mechanics']])
            st.markdown(f"**Mechanics:** {mechanics}")
            
            st.markdown("**Board Configuration:**")
            board_info = game['board']
            st.json(board_info)
            
            st.markdown("**Win Condition:**")
            st.json(game['win_condition'])
            
            st.markdown("**Rules:**")
            st.json(game['rules'])
        
        with col2:
            st.header("Quick Stats")
            
            if st.session_state.simulation_results:
                total_games = len(st.session_state.simulation_results)
                avg_length = np.mean([r['turns'] for r in st.session_state.simulation_results])
                
                st.metric("Games Simulated", total_games)
                st.metric("Avg Game Length", f"{avg_length:.1f} turns")
                
                # Show winner distribution
                winners = [r['winner'] for r in st.session_state.simulation_results if r['winner'] is not None]
                if winners:
                    winner_dist = pd.Series(winners).value_counts()
                    st.markdown("**Winner Distribution:**")
                    for player_id, count in winner_dist.items():
                        pct = (count / len(winners)) * 100
                        st.write(f"Player {player_id}: {pct:.1f}%")
            else:
                st.info("No simulations yet. Start training to see stats!")
            
            st.markdown("---")
            
            # Visualization of board
            st.subheader("Board Layout")
            if game['genre'] == 'racing':
                st.markdown(f"🏁 Linear track with {game['board'].get('length', 30)} positions")
            else:
                size = game['board'].get('size', 8)
                st.markdown(f"📐 {size}x{size} grid board")
                
                # Simple board visualization
                board_viz = np.zeros((size, size))
                fig = px.imshow(board_viz, color_continuous_scale='Viridis')
                fig.update_layout(height=300, showlegend=False)
                fig.update_xaxes(showticklabels=False)
                fig.update_yaxes(showticklabels=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: AI Performance
    with tab2:
        st.header("AI Player Performance")
        
        if st.session_state.ai_players and st.session_state.simulation_results:
            # Player statistics
            st.subheader("📈 Player Statistics")
            
            stats_data = []
            for player in st.session_state.ai_players:
                stats = player.get_stats()
                stats_data.append(stats)
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Win rate over time
            if len(st.session_state.training_history) > 1:
                st.subheader("📊 Learning Progress")
                
                # Create learning curve
                history_data = []
                for checkpoint in st.session_state.training_history:
                    game_num = checkpoint['game_number']
                    for player_stats in checkpoint['players']:
                        history_data.append({
                            'Game': game_num,
                            'Player': f"Player {player_stats['player_id']}",
                            'Win Rate': player_stats['win_rate'] * 100
                        })
                
                if history_data:
                    history_df = pd.DataFrame(history_data)
                    
                    fig = px.line(
                        history_df,
                        x='Game',
                        y='Win Rate',
                        color='Player',
                        title='Win Rate Over Time',
                        labels={'Win Rate': 'Win Rate (%)'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Q-table growth
            st.subheader("🧠 Knowledge Base Growth")
            q_table_sizes = [player.get_stats()['q_table_size'] for player in st.session_state.ai_players]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"Player {i}" for i in range(len(q_table_sizes))],
                    y=q_table_sizes,
                    text=q_table_sizes,
                    textposition='auto',
                )
            ])
            fig.update_layout(
                title="Q-Table Size (State-Action Pairs Learned)",
                xaxis_title="Player",
                yaxis_title="Number of State-Action Pairs",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Exploration rate
            st.subheader("🔍 Exploration vs Exploitation")
            epsilons = [player.epsilon for player in st.session_state.ai_players]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Exploration Rate", f"{np.mean(epsilons):.1%}")
            with col2:
                st.metric("Exploitation Rate", f"{(1 - np.mean(epsilons)):.1%}")
            
        else:
            st.info("Train the AI players to see performance metrics!")
    
    # Tab 3: Balance Analysis
    with tab3:
        st.header("Game Balance Analysis")
        
        if len(st.session_state.simulation_results) >= 10:
            analysis = st.session_state.analyzer.analyze_game_results(
                st.session_state.simulation_results,
                st.session_state.current_game['num_players']
            )
            
            # Balance grade
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Balance Grade", analysis['balance_grade'])
            with col2:
                st.metric("Fairness Score", f"{analysis['fairness_score']:.1f}/100")
            with col3:
                st.metric("Engagement Score", f"{analysis['engagement_score']:.1f}/100")
            
            st.markdown("---")
            
            # Detailed metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚖️ Win Distribution")
                win_dist = analysis['win_distribution']
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"Player {i}" for i in win_dist.keys()],
                        y=list(win_dist.values()),
                        text=[f"{v:.1f}%" for v in win_dist.values()],
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    yaxis_title="Win Percentage",
                    height=300
                )
                fig.add_hline(
                    y=100/len(win_dist), 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Perfect Balance"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("⏱️ Game Length Analysis")
                st.metric("Average Length", f"{analysis['average_game_length']:.1f} turns")
                st.metric("Standard Deviation", f"{analysis['game_length_std']:.1f} turns")
                st.metric("Complexity Score", f"{analysis['complexity_score']:.1f}/100")
            
            # Recommendations
            st.subheader("💡 AI Recommendations")
            recommendations = st.session_state.analyzer.get_recommendations(analysis)
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Game length distribution
            st.subheader("📊 Game Length Distribution")
            game_lengths = [r['turns'] for r in st.session_state.simulation_results]
            
            fig = go.Figure(data=[go.Histogram(x=game_lengths, nbinsx=20)])
            fig.update_layout(
                xaxis_title="Game Length (turns)",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Run at least 10 simulations to see balance analysis!")
    
    # Tab 4: Human Feedback
    with tab4:
        st.header("Human Feedback Integration")
        
        st.markdown("""
        Your feedback helps the AI learn what makes a good board game!
        Rate the current game design to influence future generations.
        """)
        
        game_id = st.session_state.current_game['id']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Rate This Game")
            
            fun_rating = st.slider(
                "How fun does this game look?",
                min_value=1,
                max_value=10,
                value=5,
                key=f"fun_{game_id}"
            )
            
            balance_rating = st.slider(
                "How balanced does this game seem?",
                min_value=1,
                max_value=10,
                value=5,
                key=f"balance_{game_id}"
            )
            
            complexity_rating = st.slider(
                "Is the complexity level appropriate?",
                min_value=1,
                max_value=10,
                value=5,
                key=f"complexity_{game_id}"
            )
            
            replay_rating = st.slider(
                "Would you play this game multiple times?",
                min_value=1,
                max_value=10,
                value=5,
                key=f"replay_{game_id}"
            )
            
            feedback_text = st.text_area(
                "Additional Comments",
                placeholder="What would make this game better?",
                key=f"feedback_{game_id}"
            )
            
            if st.button("Submit Feedback", type="primary"):
                st.session_state.game_ratings[game_id] = {
                    'fun': fun_rating,
                    'balance': balance_rating,
                    'complexity': complexity_rating,
                    'replay': replay_rating,
                    'comments': feedback_text,
                    'game_name': st.session_state.current_game['name']
                }
                st.success("Thank you for your feedback!")
        
        with col2:
            st.subheader("Your Rating Summary")
            
            if game_id in st.session_state.game_ratings:
                rating = st.session_state.game_ratings[game_id]
                overall = (rating['fun'] + rating['balance'] + 
                          rating['complexity'] + rating['replay']) / 4
                
                st.metric("Overall Rating", f"{overall:.1f}/10")
                st.progress(overall / 10)
                
                st.markdown("**Breakdown:**")
                st.write(f"Fun: {rating['fun']}/10")
                st.write(f"Balance: {rating['balance']}/10")
                st.write(f"Complexity: {rating['complexity']}/10")
                st.write(f"Replay Value: {rating['replay']}/10")
        
        # Show all feedback
        if st.session_state.game_ratings:
            st.markdown("---")
            st.subheader("📝 Feedback History")
            
            feedback_data = []
            for gid, rating in st.session_state.game_ratings.items():
                overall = (rating['fun'] + rating['balance'] + 
                          rating['complexity'] + rating['replay']) / 4
                feedback_data.append({
                    'Game': rating['game_name'],
                    'Overall': f"{overall:.1f}",
                    'Fun': rating['fun'],
                    'Balance': rating['balance'],
                    'Complexity': rating['complexity'],
                    'Replay': rating['replay']
                })
            
            feedback_df = pd.DataFrame(feedback_data)
            st.dataframe(feedback_df, use_container_width=True)
            
            # Feedback trends
            if len(feedback_data) > 1:
                st.subheader("📈 Feedback Trends")
                
                fig = go.Figure()
                for metric in ['Fun', 'Balance', 'Complexity', 'Replay']:
                    fig.add_trace(go.Scatter(
                        y=[d[metric] for d in feedback_data],
                        mode='lines+markers',
                        name=metric
                    ))
                
                fig.update_layout(
                    title="Rating Trends Across Games",
                    xaxis_title="Game Number",
                    yaxis_title="Rating (1-10)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎲 Self-Learning Board Game Designer - Demonstrating AI's Creative Potential</p>
    <p>Powered by Reinforcement Learning (Q-Learning) and Autonomous Game Generation</p>
</div>
""", unsafe_allow_html=True)
