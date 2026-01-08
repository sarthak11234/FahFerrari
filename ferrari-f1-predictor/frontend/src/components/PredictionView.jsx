import React from 'react';
import { Target, BarChart2, Award } from 'lucide-react';

const PredictionView = ({ predictions }) => {
    if (!predictions) return <div>Generating predictions...</div>;

    return (
        <div className="prediction-view">
            <div className="card" style={{ borderLeft: '4px solid var(--ferrari-yellow)' }}>
                <h2><Target /> 2026 Championship Forecast</h2>
                <div className="dashboard-grid">
                    <div>
                        <h3>Predicted Points</h3>
                        <div className="stat-value" style={{ color: 'var(--ferrari-yellow)' }}>
                            {predictions.predicted_points}
                        </div>
                        <div className="text-muted">
                            CI: [{predictions.confidence_interval[0]} - {predictions.confidence_interval[1]}]
                        </div>
                    </div>
                    <div>
                        <h3>Predicted Position</h3>
                        <div className="stat-value">P{predictions.predicted_position}</div>
                    </div>
                </div>
            </div>

            <div className="dashboard-grid" style={{ marginTop: '1.5rem' }}>
                <div className="card">
                    <h2><Award /> Expected Performance</h2>
                    <div className="stat-item">
                        <span>Expected Wins</span>
                        <div className="stat-value" style={{ fontSize: '1.5rem' }}>
                            {predictions.predicted_wins}
                        </div>
                    </div>
                    <div className="stat-item">
                        <span>Expected Podiums</span>
                        <div className="stat-value" style={{ fontSize: '1.5rem' }}>
                            {predictions.predicted_podiums}
                        </div>
                    </div>
                </div>

                <div className="card">
                    <h2><BarChart2 /> Model Consensus</h2>
                    <div className="models-list">
                        {Object.entries(predictions.individual_predictions || {}).map(([model, value]) => (
                            <div key={model} style={{ marginBottom: '1rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <span>{model.replace('_', ' ').toUpperCase()}</span>
                                    <span>{value} pts</span>
                                </div>
                                <div className="prob-bar">
                                    <div
                                        className="prob-fill"
                                        style={{ width: `${(value / 800) * 100}%` }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* New Detailed Sections */}
            {predictions.driver_standings && (
                <div className="dashboard-grid" style={{ marginTop: '1.5rem' }}>
                    <div className="card">
                        <h2>🏎️ Drivers Championship</h2>
                        <div style={{ marginTop: '1rem' }}>
                            {Object.entries(predictions.driver_standings).map(([driver, stats]) => (
                                <div key={driver} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', padding: '0.5rem', background: '#333', borderRadius: '8px' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                        <div style={{ width: '30px', height: '30px', borderRadius: '50%', background: driver === 'leclerc' ? '#ff2800' : '#444', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                            {driver[0].toUpperCase()}
                                        </div>
                                        <span style={{ textTransform: 'capitalize', fontWeight: 'bold' }}>{driver}</span>
                                    </div>
                                    <div style={{ textAlign: 'right' }}>
                                        <div style={{ fontWeight: 'bold', fontSize: '1.2rem' }}>{stats.points} pts</div>
                                        <div style={{ fontSize: '0.8rem', color: '#888' }}>{stats.wins} Wins | {stats.podiums} Podiums</div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="card">
                        <h2>🏆 Constructors Standings</h2>
                        <div style={{ marginTop: '1rem' }}>
                            {predictions.constructors_standings?.slice(0, 5).map((team) => (
                                <div key={team.team} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem', borderBottom: '1px solid #444' }}>
                                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                                        <span style={{ color: '#888' }}>{team.position}.</span>
                                        <span style={{ fontWeight: team.team === 'Ferrari' ? 'bold' : 'normal', color: team.team === 'Ferrari' ? 'var(--ferrari-yellow)' : 'white' }}>
                                            {team.team}
                                        </span>
                                    </div>
                                    <span>{team.points} pts</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {predictions.race_predictions && (
                <div className="card" style={{ marginTop: '1.5rem' }}>
                    <h2>📅 Race-by-Race Forecast</h2>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
                        {predictions.race_predictions.map((race) => (
                            <div key={race.round} style={{ background: '#333', padding: '1rem', borderRadius: '8px', border: race.race_position === 'P1' ? '1px solid var(--ferrari-red)' : '1px solid #444' }}>
                                <div style={{ fontSize: '0.8rem', color: '#888', display: 'flex', justifyContent: 'space-between' }}>
                                    <span>Round {race.round}</span>
                                    <span>{race.performance_score}</span>
                                </div>
                                <div style={{ fontWeight: 'bold', marginBottom: '0.5rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{race.circuit}</div>

                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.5rem' }}>
                                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                                        <span style={{ fontSize: '1.5rem', fontWeight: 'bold', color: race.race_position === 'P1' ? 'var(--ferrari-yellow)' : 'white' }}>
                                            {race.race_position}
                                        </span>
                                        <span style={{ fontSize: '0.75rem', color: '#aaa' }}>Position</span>
                                    </div>
                                    <div style={{ textAlign: 'right' }}>
                                        <span style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>{race.predicted_points}</span>
                                        <div style={{ fontSize: '0.75rem', color: '#aaa' }}>Points</div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default PredictionView;
