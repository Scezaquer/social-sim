let simData = null;
let degreeChart = null;
let powerlawChart = null;
let surveyChart = null;
let opinionChangesChart = null;
let simulationGraph = null;
let currentStep = 0;
let maxStep = 0;
let surveySteps = [];
let selectedNode = null;

document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            simData = JSON.parse(e.target.result);
            processData();
            initVisualizations();
        } catch (err) {
            console.error("Error parsing JSON:", err);
            alert("Invalid JSON file.");
        }
    };
    reader.readAsText(file);
});

function processData() {
    // Calculate degrees
    const degrees = new Array(simData.nodes.length).fill(0);
    simData.edges.forEach(edge => {
        degrees[edge.source]++;
        degrees[edge.target]++;
    });
    simData.nodes.forEach((node, i) => {
        node.degree = degrees[i];
    });

    // Find max step
    maxStep = 0;
    if (simData.survey_results.length > 0) {
        maxStep = Math.max(maxStep, ...simData.survey_results.map(s => s.step));
    }
    if (simData.observations.length > 0) {
        maxStep = Math.max(maxStep, ...simData.observations.map(o => o.step));
    }
    
    surveySteps = simData.survey_results.map(s => s.step).sort((a, b) => a - b);

    // Setup slider
    const slider = document.getElementById('timeSlider');
    slider.max = surveySteps.length > 0 ? surveySteps.length - 1 : 0;
    slider.value = 0;
    currentStep = surveySteps.length > 0 ? surveySteps[0] : 0;
    document.getElementById('timeValue').innerText = currentStep;

    slider.addEventListener('input', function() {
        const idx = parseInt(this.value);
        currentStep = surveySteps[idx];
        document.getElementById('timeValue').innerText = currentStep;
        updateGraph();
    });

    document.getElementById('toggleLatestVoteColor').addEventListener('change', updateGraph);
    document.getElementById('toggleChangedMinds').addEventListener('change', updateGraph);
    document.getElementById('toggleOpinionChangeColor').addEventListener('change', updateGraph);
    document.getElementById('toggleNodeSize').addEventListener('change', updateGraph);
    document.getElementById('toggleNodeSizeViews').addEventListener('change', updateGraph);
    document.getElementById('toggleNewsLines').addEventListener('change', updateSurveyChart);
    document.querySelectorAll('input[name="surveyYAxis"]').forEach(radio => {
        radio.addEventListener('change', updateSurveyChart);
    });
}

function initVisualizations() {
    drawDegreeDistribution();
    drawPowerlawDistribution();
    drawSurveyChart();
    drawOpinionChangesDistribution();
    drawMessageReach();
    initGraph();
    updateGraph();
}

function drawDegreeDistribution() {
    const degrees = simData.nodes.map(n => n.degree);
    const maxDegree = Math.max(...degrees);
    const bins = new Array(maxDegree + 1).fill(0);
    degrees.forEach(d => bins[d]++);

    const ctx = document.getElementById('degreeChart').getContext('2d');
    if (degreeChart) degreeChart.destroy();

    degreeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Array.from({length: maxDegree + 1}, (_, i) => i),
            datasets: [{
                label: 'Number of Nodes',
                data: bins,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'Degree' } },
                y: { title: { display: true, text: 'Count' }, type: 'logarithmic' }
            }
        }
    });
}

function drawPowerlawDistribution() {
    const degrees = simData.nodes.map(n => n.degree);
    const maxDegree = Math.max(...degrees);
    const bins = new Array(maxDegree + 1).fill(0);
    degrees.forEach(d => bins[d]++);

    const scatterData = [];
    for (let i = 1; i <= maxDegree; i++) {
        if (bins[i] > 0) {
            scatterData.push({ x: i, y: bins[i] });
        }
    }

    const ctx = document.getElementById('powerlawChart').getContext('2d');
    if (powerlawChart) powerlawChart.destroy();

    powerlawChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Degree Distribution',
                data: scatterData,
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { 
                    title: { display: true, text: 'Degree (log)' },
                    type: 'logarithmic'
                },
                y: { 
                    title: { display: true, text: 'Count (log)' },
                    type: 'logarithmic'
                }
            }
        }
    });
}

function drawSurveyChart() {
    if (simData.survey_results.length > 0) {
        document.getElementById('surveyQuestion').innerText = "Question: " + simData.survey_results[0].question;
    }

    const ctx = document.getElementById('surveyChart').getContext('2d');
    if (surveyChart) surveyChart.destroy();

    const steps = simData.survey_results.map(s => s.step);
    const yAxisMode = document.querySelector('input[name="surveyYAxis"]:checked').value;
    
    // Get all unique answers
    const allAnswers = new Set();
    simData.survey_results.forEach(s => {
        Object.values(s.results).forEach(ans => allAnswers.add(ans));
    });
    
    const datasets = Array.from(allAnswers).map((ans, i) => {
        const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];
        const data = simData.survey_results.map(s => {
            let count = 0;
            const totalResponses = Object.keys(s.results).length;
            Object.values(s.results).forEach(val => {
                if (val === ans) count++;
            });
            
            let yValue = count;
            if (yAxisMode === 'percentage' && totalResponses > 0) {
                yValue = (count / totalResponses) * 100;
            }
            
            return { x: s.step, y: yValue };
        });
        return {
            label: ans,
            data: data,
            borderColor: colors[i % colors.length],
            fill: false,
            tension: 0.1
        };
    });

    const annotations = {};
    simData.news_posts.forEach((post, i) => {
        annotations[`line${i}`] = {
            type: 'line',
            xMin: post.step,
            xMax: post.step,
            borderColor: 'rgba(255, 99, 132, 0.5)',
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
                content: 'News',
                display: true,
                position: 'start'
            }
        };
    });

    let yAxisMax = undefined;
    if (yAxisMode === 'percentage') {
        yAxisMax = 100;
    } else {
        // For absolute count, max is the total number of nodes that answered
        yAxisMax = simData.survey_results.length > 0 ? Object.keys(simData.survey_results[0].results).length : undefined;
    }

    surveyChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                annotation: {
                    annotations: document.getElementById('toggleNewsLines').checked ? annotations : {}
                }
            },
            scales: {
                x: { 
                    type: 'linear',
                    title: { display: true, text: 'Step' },
                    ticks: {
                        stepSize: steps.length > 1 ? steps[1] - steps[0] : 100
                    }
                },
                y: { 
                    title: { display: true, text: yAxisMode === 'percentage' ? 'Percentage (%)' : 'Count' },
                    min: 0,
                    max: yAxisMax
                }
            }
        }
    });
}

function updateSurveyChart() {
    drawSurveyChart();
}

function drawOpinionChangesDistribution() {
    const changesCount = new Array(simData.nodes.length).fill(0);
    
    for (let i = 1; i < simData.survey_results.length; i++) {
        const prevSurvey = simData.survey_results[i - 1].results;
        const currSurvey = simData.survey_results[i].results;
        
        simData.nodes.forEach((node, idx) => {
            if (node.type !== 'NewsSource' && prevSurvey[node.name] && currSurvey[node.name]) {
                if (prevSurvey[node.name] !== currSurvey[node.name]) {
                    changesCount[idx]++;
                }
            }
        });
    }

    const maxChanges = Math.max(...changesCount);
    const bins = new Array(maxChanges + 1).fill(0);
    changesCount.forEach((count, idx) => {
        if (simData.nodes[idx].type !== 'NewsSource') {
            bins[count]++;
        }
    });

    const ctx = document.getElementById('opinionChangesChart').getContext('2d');
    if (opinionChangesChart) opinionChangesChart.destroy();

    opinionChangesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Array.from({length: maxChanges + 1}, (_, i) => i),
            datasets: [{
                label: 'Number of Nodes',
                data: bins,
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'Number of Opinion Changes' } },
                y: { title: { display: true, text: 'Count' } }
            }
        }
    });
}

function drawMessageReach() {
    const tbody = document.querySelector('#messageReachTable tbody');
    tbody.innerHTML = '';

    // Calculate views per thread
    const threadViews = {};
    simData.observations.forEach(obs => {
        threadViews[obs.thread_id] = (threadViews[obs.thread_id] || 0) + 1;
    });

    // Get first message of each thread
    const threadInfo = [];
    simData.threads.forEach(t => {
        if (t.messages && t.messages.length > 0) {
            threadInfo.push({
                id: t.id,
                author: t.messages[0].role,
                content: t.messages[0].content.substring(0, 50) + '...',
                views: threadViews[t.id] || 0
            });
        }
    });

    threadInfo.sort((a, b) => b.views - a.views);

    threadInfo.slice(0, 50).forEach(info => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${info.id}</td>
            <td>${info.author}</td>
            <td title="${info.content}" class="message-cell" onclick="showThreadDetails(${info.id})">${info.content}</td>
            <td>${info.views}</td>
        `;
        tbody.appendChild(tr);
    });
}

function showThreadDetails(threadId) {
    const thread = simData.threads.find(t => t.id === threadId);
    if (!thread) return;
    
    // Calculate total thread views
    let totalThreadViews = 0;
    simData.observations.forEach(obs => {
        if (obs.thread_id === threadId && obs.step <= currentStep) {
            totalThreadViews++;
        }
    });

    let content = `<h3>Thread ${threadId} (Seen ${totalThreadViews} times)</h3><div style="max-height: 400px; overflow-y: auto;">`;
    thread.messages.forEach(m => {
        if (m.step <= currentStep) {
            // Calculate views for this specific message
            let messageViews = 0;
            simData.observations.forEach(obs => {
                if (obs.thread_id === threadId && obs.step >= m.step && obs.step <= currentStep) {
                    messageViews++;
                }
            });
            content += `<div class="message-item"><strong>${m.role}</strong> (Step ${m.step || 'N/A'}) [Seen ${messageViews} times]: ${m.content}</div>`;
        }
    });
    content += `</div>`;
    
    // Create a modal or use the right panel
    const infoDiv = document.getElementById('node-info');
    infoDiv.innerHTML = content;
    document.getElementById('node-threads').innerHTML = '';
    document.getElementById('node-messages').innerHTML = '';
}

function initGraph() {
    const container = document.getElementById('graph-container');
    container.innerHTML = '';
    const width = container.clientWidth;
    const height = container.clientHeight || 500;

    const svg = d3.select('#graph-container').append('svg')
        .attr('width', width)
        .attr('height', height)
        .call(d3.zoom().on('zoom', (event) => {
            g.attr('transform', event.transform);
        }));

    svg.on('click', (event) => {
        if (event.target.tagName === 'svg') {
            selectNode(null);
        }
    });

    const g = svg.append('g');

    simulationGraph = d3.forceSimulation(simData.nodes)
        .force('link', d3.forceLink(simData.edges).id(d => d.id).distance(30))
        .force('charge', d3.forceManyBody().strength(-50))
        .force('center', d3.forceCenter(width / 2, height / 2));

    const link = g.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(simData.edges)
        .enter().append('line')
        .attr('class', 'link');

    const node = g.append('g')
        .attr('class', 'nodes')
        .selectAll('circle')
        .data(simData.nodes)
        .enter().append('circle')
        .attr('class', 'node')
        .attr('r', d => d.type === 'NewsSource' ? 8 : 5)
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended))
        .on('click', (event, d) => selectNode(d));

    node.append('title')
        .text(d => d.name);

    simulationGraph.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
    });

    function dragstarted(event, d) {
        if (!event.active) simulationGraph.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulationGraph.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

function updateGraph() {
    if (!simData || !simulationGraph) return;

    const highlightChanged = document.getElementById('toggleChangedMinds').checked;
    const colorByOpinionChange = document.getElementById('toggleOpinionChangeColor').checked;
    const colorByLatestVote = document.getElementById('toggleLatestVoteColor').checked;
    const sizeByMessages = document.getElementById('toggleNodeSize').checked;
    const sizeByViews = document.getElementById('toggleNodeSizeViews').checked;
    
    // Get survey results for current step
    const currentSurvey = simData.survey_results.find(s => s.step === currentStep);
    
    // Get survey results for previous step
    const currentIdx = surveySteps.indexOf(currentStep);
    const prevStep = currentIdx > 0 ? surveySteps[currentIdx - 1] : null;
    const prevSurvey = prevStep !== null ? simData.survey_results.find(s => s.step === prevStep) : null;

    // Calculate opinion changes up to current step
    const changesCount = new Array(simData.nodes.length).fill(0);
    if (colorByOpinionChange) {
        for (let i = 1; i <= currentIdx; i++) {
            const pSurvey = simData.survey_results[i - 1].results;
            const cSurvey = simData.survey_results[i].results;
            
            simData.nodes.forEach((node, idx) => {
                if (node.type !== 'NewsSource' && pSurvey[node.name] && cSurvey[node.name]) {
                    if (pSurvey[node.name] !== cSurvey[node.name]) {
                        changesCount[idx]++;
                    }
                }
            });
        }
    }
    const maxChanges = Math.max(...changesCount, 1);

    // Calculate messages written up to current step
    const messagesWritten = new Array(simData.nodes.length).fill(0);
    if (sizeByMessages) {
        simData.threads.forEach(t => {
            if (t.messages) {
                t.messages.forEach(m => {
                    if (m.step <= currentStep) {
                        const nodeIdx = simData.nodes.findIndex(n => n.name === m.role);
                        if (nodeIdx !== -1) {
                            messagesWritten[nodeIdx]++;
                        }
                    }
                });
            }
        });
    }
    const maxMessages = Math.max(...messagesWritten, 1);

    // Calculate cumulative views up to current step
    const cumulativeViews = new Array(simData.nodes.length).fill(0);
    if (sizeByViews) {
        const threadMap = new Map();
        simData.threads.forEach(t => threadMap.set(t.id, t));
        const nodeIdxMap = new Map();
        simData.nodes.forEach((n, idx) => nodeIdxMap.set(n.name, idx));
        
        simData.observations.forEach(obs => {
            if (obs.step <= currentStep) {
                const thread = threadMap.get(obs.thread_id);
                if (thread && thread.messages) {
                    thread.messages.forEach(m => {
                        if (m.step <= obs.step) {
                            const nodeIdx = nodeIdxMap.get(m.role);
                            if (nodeIdx !== undefined) {
                                cumulativeViews[nodeIdx]++;
                            }
                        }
                    });
                }
            }
        });
    }
    const maxViews = Math.max(...cumulativeViews, 1);

    // Get all unique answers to assign colors
    const allAnswers = Array.from(new Set(simData.survey_results.flatMap(s => Object.values(s.results))));
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10).domain(allAnswers);

    d3.selectAll('.node')
        .attr('r', (d, i) => {
            if (d.type === 'NewsSource') return 8;
            if (sizeByViews) {
                return 3 + (cumulativeViews[i] / maxViews) * 15;
            } else if (sizeByMessages) {
                return 3 + (messagesWritten[i] / maxMessages) * 15;
            }
            return 5;
        })
        .attr('fill', (d, i) => {
            if (d.type === 'NewsSource') return '#000'; // Black for news sources
            
            let nodeColor;
            
            if (colorByLatestVote && currentSurvey && currentSurvey.results[d.name]) {
                nodeColor = d3.color(colorScale(currentSurvey.results[d.name]));
            } else {
                nodeColor = d3.color('#1f77b4'); // Default blue
            }
            
            if (colorByOpinionChange) {
                const changeRatio = changesCount[i] / maxChanges;
                // Darker if changed more, lighter if changed less
                const k = changeRatio * 3 - 1;
                if (k > 0) {
                    nodeColor = nodeColor.darker(k);
                } else {
                    nodeColor = nodeColor.brighter(-k);
                }
            }
            
            if (highlightChanged) {
                if (!prevSurvey || !prevSurvey.results[d.name] || !currentSurvey || !currentSurvey.results[d.name]) {
                    nodeColor = d3.color('#eee');
                } else {
                    const prevAns = prevSurvey.results[d.name];
                    const currentAns = currentSurvey.results[d.name];
                    if (currentAns === prevAns) {
                        nodeColor = d3.color('#eee'); // Gray out if not changed
                    }
                }
            }
            
            return nodeColor.formatHex();
        })
        .attr('stroke', d => d === selectedNode ? '#000' : '#fff')
        .attr('stroke-width', d => d === selectedNode ? 3 : 1.5)
        .attr('opacity', d => selectedNode && d !== selectedNode ? 0.3 : 1);
        
    d3.selectAll('.link')
        .attr('stroke-opacity', d => {
            if (!selectedNode) return 0.6;
            return (d.source === selectedNode || d.target === selectedNode) ? 1 : 0.1;
        });
        
    updateNodeDetails(selectedNode);
}

function selectNode(node) {
    selectedNode = node;
    updateGraph();
}

function updateNodeDetails(node) {
    const infoDiv = document.getElementById('node-info');
    if (!node) {
        infoDiv.innerHTML = 'Select a node to see details.';
        document.getElementById('node-threads').innerHTML = '';
        document.getElementById('node-messages').innerHTML = '';
        return;
    }

    infoDiv.innerHTML = `
        <strong>Name:</strong> ${node.name}<br>
        <strong>Type:</strong> ${node.type}<br>
        <strong>Model ID:</strong> ${node.model_id || 'N/A'}<br>
        <strong>Degree:</strong> ${node.degree}
    `;

    // Find threads seen by this node up to currentStep
    const seenThreads = new Set();
    simData.observations.forEach(obs => {
        if (obs.entity_name === node.name && obs.step <= currentStep) {
            seenThreads.add(obs.thread_id);
        }
    });

    const threadsDiv = document.getElementById('node-threads');
    threadsDiv.innerHTML = `<h3>Threads Seen (${seenThreads.size})</h3>`;
    const threadsList = document.createElement('div');
    threadsList.style.maxHeight = '300px';
    threadsList.style.overflowY = 'auto';
    
    Array.from(seenThreads).forEach(tid => {
        const div = document.createElement('div');
        div.className = 'thread-item clickable';
        div.innerText = `Thread ${tid}`;
        div.onclick = () => showThreadDetails(tid);
        threadsList.appendChild(div);
    });
    threadsDiv.appendChild(threadsList);

    // Find messages written by this node up to currentStep
    const writtenMessages = [];
    simData.threads.forEach(t => {
        if (t.messages) {
            t.messages.forEach(m => {
                if (m.role === node.name && m.step <= currentStep) {
                    // Calculate how many times this specific message was seen
                    let messageViews = 0;
                    simData.observations.forEach(obs => {
                        if (obs.thread_id === t.id && obs.step >= m.step && obs.step <= currentStep) {
                            messageViews++;
                        }
                    });
                    writtenMessages.push({thread_id: t.id, content: m.content, step: m.step, views: messageViews});
                }
            });
        }
    });

    const msgsDiv = document.getElementById('node-messages');
    msgsDiv.innerHTML = `<h3>Messages Written (${writtenMessages.length})</h3>`;
    const msgsList = document.createElement('div');
    msgsList.style.maxHeight = '300px';
    msgsList.style.overflowY = 'auto';
    
    let totalViews = 0;
    writtenMessages.sort((a, b) => b.step - a.step).forEach(m => {
        totalViews += m.views;
        const div = document.createElement('div');
        div.className = 'message-item clickable';
        div.innerHTML = `<strong>Step ${m.step} (Thread ${m.thread_id}) [Seen ${m.views} times]:</strong> ${m.content}`;
        div.onclick = () => showThreadDetails(m.thread_id);
        msgsList.appendChild(div);
    });
    msgsDiv.appendChild(msgsList);
    
    infoDiv.innerHTML += `<br><strong>Total Views:</strong> ${totalViews}`;
}

function focusPanel(panelId) {
    const panels = document.querySelectorAll('.focusable');
    panels.forEach(p => {
        if (p.id === panelId) {
            p.classList.add('focused');
            document.getElementById('center-panel').appendChild(p);
        } else {
            p.classList.remove('focused');
            document.getElementById('left-panel').appendChild(p);
        }
    });
    
    // Resize charts if they exist
    if (degreeChart) degreeChart.resize();
    if (powerlawChart) powerlawChart.resize();
    if (surveyChart) surveyChart.resize();
    if (opinionChangesChart) opinionChangesChart.resize();
    
    // Resize D3 graph if it's the focused panel
    if (panelId === 'graph-panel' && simulationGraph) {
        const container = document.getElementById('graph-container');
        const width = container.clientWidth;
        const height = container.clientHeight || 500;
        
        d3.select('#graph-container svg')
            .attr('width', width)
            .attr('height', height);
            
        simulationGraph.force('center', d3.forceCenter(width / 2, height / 2));
        simulationGraph.alpha(0.3).restart();
    }
}
