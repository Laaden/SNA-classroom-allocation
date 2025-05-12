import { Link } from "react-router-dom";
import "../styles/classforge.css";

export default function HomePage() {
  return (
    <div>
      {/* Header Section */}
        <header>
            <div className="header-container">
                <div className="logo">
                    <a href="index.html" className="logo-link">ClassForge</a>
                </div>
                <nav className="navbar">
                    <ul className="nav-list">
                        <li><Link to="/home">Home</Link></li>
                        <li><Link to="/result">Dashboard</Link></li>
                        <li><Link to="/weights">Allocation</Link></li>
                        <li><a href="contact.html">Contact</a></li>
                    </ul>
                </nav>
            </div>
        </header>

        {/* Hero Section */}
        <section id="hero">
            <div className="hero-content">
                <h1>Optimize Your Classroom With AI</h1>
                <p>Revolutionize class allocations with advanced social network analysis and multi-objective optimization.</p>
                <Link to="/weights" className="cta-button">Discover How</Link>
            </div>
        </section>

            {/* Services Section */}
        <section id="services">
            <div className="container">
                <h2>Services</h2>
                <div className="service-cards">
                    <div className="card">
                        <h3>AI-Powered Optimization</h3>
                        <p>Generate optimal classroom groupings by balancing academic performance, social ties, and overall well-being.</p>
                    </div>
                    <div class="card">
                        <h3>Real-Time Dashboard</h3>
                        <p>Visualize student interactions, allocation scenarios, and performance metrics instantly.</p>
                    </div>
                    <div class="card">
                        <h3>User-Centric Control</h3>
                        <p>Enjoy manual override capabilities with our intuitive drag-and-drop interface for ultimate flexibility.</p>
                    </div>
                </div>
            </div>
        </section>

        {/* Contact Section */}
        <section id="contact">
            <div class="container">
                <h2>Get in Touch</h2>
                <p>Questions or feedback? Contact us today and let us help you optimize your classrooms.</p>
                <form action="#" method="post">
                    <input type="text" placeholder="Your Name" required />
                    <input type="email" placeholder="Your Email" required />
                    <input type="tel" placeholder="Your Phone" required />
                    <textarea placeholder="Your Message" required></textarea>
                    <button type="submit">Submit</button>
                </form>
            </div>
        </section>

        {/* Footer */}
        <footer>
            <div className="container">
                <p>&copy; 2025 ClassForge. All rights reserved.</p>
            </div>
        </footer>
      </div>
  );
}
