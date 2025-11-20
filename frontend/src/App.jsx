// src/App.jsx
import { useState } from 'react'
import './App.css'

function App() {
  // 1. State to hold form data
  const [formData, setFormData] = useState({
    source: '',
    destination: '',
    startDate: '',
    endDate: '',
    people: 2,
    budget: 15000, // Default value
    vehicle: 'rental' // Default value
  })

  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState(null)

  // 2. Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  // 3. Send data to FastAPI Backend
  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setResponse(null)

    try {
      const res = await fetch('http://127.0.0.1:8000/api/generate-plan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      const data = await res.json()
      setResponse(data) // Save the backend response
    } catch (error) {
      console.error("Error:", error)
      alert("Failed to connect to the Bots. Is the backend running?")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <div className="card">
        <h1>✈️ Trip Builder AI</h1>
        
        <form onSubmit={handleSubmit}>
          
          {/* Row 1: Locations */}
          <div className="form-group">
            <label>From (Source)</label>
            <input 
              type="text" 
              name="source" 
              placeholder="e.g., Mumbai" 
              value={formData.source} 
              onChange={handleChange} 
              required 
            />
          </div>

          <div className="form-group">
            <label>To (Destination)</label>
            <input 
              type="text" 
              name="destination" 
              placeholder="e.g., Goa" 
              value={formData.destination} 
              onChange={handleChange} 
              required 
            />
          </div>

          {/* Row 2: Dates */}
          <div style={{ display: 'flex', gap: '10px' }}>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Start Date</label>
              <input 
                type="date" 
                name="startDate" 
                value={formData.startDate} 
                onChange={handleChange} 
                required 
              />
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label>End Date</label>
              <input 
                type="date" 
                name="endDate" 
                value={formData.endDate} 
                onChange={handleChange} 
                required 
              />
            </div>
          </div>

          {/* Row 3: Vehicle & People */}
          <div style={{ display: 'flex', gap: '10px' }}>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Vehicle</label>
              <select name="vehicle" value={formData.vehicle} onChange={handleChange}>
                <option value="rental">Rental Car / Taxi</option>
                <option value="own">Own Vehicle</option>
                <option value="public">Public Transport</option>
              </select>
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Travelers</label>
              <input 
                type="number" 
                name="people" 
                min="1" 
                value={formData.people} 
                onChange={handleChange} 
                required 
              />
            </div>
          </div>

          {/* Row 4: Budget Scale */}
          <div className="form-group">
            <label>Budget Per Person (Scale)</label>
            <div className="budget-scale-container">
              <input 
                type="range" 
                name="budget" 
                min="1000" 
                max="50000" 
                step="500" 
                value={formData.budget} 
                onChange={handleChange} 
              />
              <div className="budget-value">₹ {formData.budget}</div>
            </div>
          </div>

          {/* Submit Button */}
          <button type="submit" className="btn-search" disabled={loading}>
            {loading ? '🤖 Bots are Planning...' : 'Search Packages'}
          </button>

        </form>

        {/* Results Display */}
        {response && (
          <div className="result-box">
            <h3>✅ Plan Generated!</h3>
            <p><strong>Weather:</strong> {response.data.destination_analysis.weather}</p>
            <p><strong>Top Spots:</strong> {response.data.destination_analysis.top_spots?.join(', ')}</p>
            <p style={{fontSize: '0.8rem', color: '#666'}}>Check console for full JSON response</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default App