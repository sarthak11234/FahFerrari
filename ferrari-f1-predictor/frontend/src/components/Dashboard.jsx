import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Trophy, Flag, AlertTriangle, TrendingUp } from 'lucide-react';

const Dashboard = ({ data }) => {
    if (!data || data.length === 0) return <div>No data available</div>;

    const latestStats = data[data.length - 1] || {};
    const totalWins = data.reduce((acc, curr) => acc + (curr.wins || 0), 0);
    const totalPodiums = data.reduce((acc, curr) => acc + (curr.podiums || 0), 0);

    return (
        <div className="dashboard">
            <div className="dashboard-grid">
                <div className="card">
                    <h2><Trophy size={20} /> Total Wins (2019-2025)</h2>
                    <div className="stat-value">{totalWins}</div>
                </div>
                <div className="card">
                    <h2><Flag size={20} /> Podiums</h2>
                    <div className="stat-value">{totalPodiums}</div>
                </div>
                <div className="card">
                    <h2><TrendingUp size={20} /> Latest Points</h2>
                    <div className="stat-value">{latestStats.total_points || 0}</div>
                </div>
            </div>

            <div className="card" style={{ marginTop: '1.5rem' }}>
                <h2>Championship Points Trend</h2>
                <div style={{ height: 300, width: '100%' }}>
                    <ResponsiveContainer>
                        <LineChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                            <XAxis dataKey="year" stroke="#888" />
                            <YAxis stroke="#888" />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#333', border: 'none' }}
                                itemStyle={{ color: '#fff' }}
                            />
                            <Line
                                type="monotone"
                                dataKey="total_points"
                                stroke="#ff2800"
                                strokeWidth={3}
                                dot={{ fill: '#ff2800' }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
